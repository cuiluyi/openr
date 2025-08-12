import warnings
import torch
import numpy as np

from typing import Any, Union, Optional, Tuple
from trl import DataCollatorForCompletionOnlyLM
from transformers import DataCollatorForLanguageModeling


class DataCollatorForTagOnly(DataCollatorForLanguageModeling):
    def __init__(
        self,
        tag_tokens: Tuple[str, ...],
        *args,
        mlm: bool = False,
        ignore_index: int = -100,
        padding_free: bool = False,
        **kwargs,
    ):
        super().__init__(*args, mlm=mlm, **kwargs)

        self.tag_tokens = tag_tokens

        self.tag_token_ids: list[int] = self.tokenizer.convert_tokens_to_ids(self.tag_tokens)

        if not self.mlm and self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            warnings.warn(
                "The pad_token_id and eos_token_id values of this tokenizer are identical. "
                "If you are planning for multi-turn training, "
                "it can result in the model continuously generating questions and answers without eos token. "
                "To avoid this, set the pad_token_id to a different value.",
                UserWarning,
            )

        self.ignore_index = ignore_index
        self.padding_free = padding_free

    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        batch = super().torch_call(examples)

        # convert tag_tokens to tensor
        tag_token_ids_tensor = torch.tensor(self.tag_token_ids, device=batch["labels"].device)

        for i in range(len(examples)):
            labels = batch["labels"][i]

            # mask=True means we want to keep the tag tokens
            mask = torch.isin(labels, tag_token_ids_tensor)

            # Set positions not in tag_token_ids to ignore_index
            labels[~mask] = self.ignore_index

            batch["labels"][i] = labels

        if self.padding_free:
            # remove padding, `attention_mask` and add `position_ids`
            attn_mask = batch.pop("attention_mask")
            batch["input_ids"] = batch["input_ids"][attn_mask.bool()].unsqueeze(0)
            batch["position_ids"] = attn_mask.cumsum(1)[attn_mask.bool()].unsqueeze(0) - 1
            batch["labels"] = batch["labels"][attn_mask.bool()].unsqueeze(0)
            batch["labels"][batch["position_ids"] == 0] = self.ignore_index

            # Calculate cumulative sequence lengths for queries and keys to prevent graph breaks during further computations.
            flattened_position_ids = batch["position_ids"].flatten()
            indices_q = torch.arange(
                flattened_position_ids.size(0), device=flattened_position_ids.device, dtype=torch.int32
            )
            batch["cu_seq_lens_q"] = torch.cat(
                (
                    indices_q[flattened_position_ids == 0],
                    torch.tensor(
                        flattened_position_ids.size(), device=flattened_position_ids.device, dtype=torch.int32
                    ),
                )
            ).unsqueeze(0)
            batch["cu_seq_lens_k"] = batch["cu_seq_lens_q"]

            # Determine maximum sequence lengths to prevent graph breaks during further computations.
            batch["max_length_k"] = torch.tensor([flattened_position_ids.max().item() + 1])
            batch["max_length_q"] = batch["max_length_k"]

        return batch


class DataCollatorForCompletionOnly(DataCollatorForLanguageModeling):
    """
    Data collator used for completion tasks. It ensures that all the tokens of the labels are set to an 'ignore_index'
    when they do not come from the assistant. This ensure that the loss is only
    calculated on the completion made by the assistant.

    Args:
        response_template (`Union[str, list[int]]`): the template form that indicates the start of the response, typically something like
            '### Response:\n'. It can also be passed as tokenized ids, which can be useful when using a tokenizer that encodes the response
            differently if it does not have proper context.
        mlm (`bool`, *optional*, defaults to `False`): Whether to use masked language modeling in the underlying
            `DataCollatorForLanguageModeling` class. Note that this option currently has no effect but is present
             for flexibility and backwards-compatibility.
        ignore_index (`int`, *optional*, defaults to `-100`):
            The index to use to ignore the initial tokens with
    """

    def __init__(
        self,
        response_template: Union[str, list[int]],
        *args,
        mlm: bool = False,
        ignore_index: int = -100,
        padding_free: bool = False,
        **kwargs,
    ):
        super().__init__(*args, mlm=mlm, **kwargs)

        self.response_template = response_template
        if isinstance(response_template, str):
            # The user provides a string, must tokenize
            self.response_token_ids = self.tokenizer.encode(self.response_template, add_special_tokens=False)
        else:
            # The user already provides the token ids
            self.response_token_ids = response_template

        if not self.mlm and self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            warnings.warn(
                "The pad_token_id and eos_token_id values of this tokenizer are identical. "
                "If you are planning for multi-turn training, "
                "it can result in the model continuously generating questions and answers without eos token. "
                "To avoid this, set the pad_token_id to a different value.",
                UserWarning,
            )

        self.ignore_index = ignore_index
        self.padding_free = padding_free

    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        batch = super().torch_call(examples)
        processed_record = {key: [] for key in batch.keys()}

        for i in range(len(examples)):
            response_token_ids_start_idx = None

            for idx in np.where(batch["labels"][i] == self.response_token_ids[0])[0]:
                if (
                    self.response_token_ids
                    == batch["labels"][i][idx : idx + len(self.response_token_ids)].tolist()
                ):
                    response_token_ids_start_idx = idx

                if response_token_ids_start_idx is None:
                    warnings.warn(
                        f"Could not find response key `{self.response_template}` in the following instance: "
                        f"{self.tokenizer.decode(batch['input_ids'][i])}. This instance will be ignored in loss "
                        "calculation. Note, if this happens often, consider increasing the `max_length`.",
                        UserWarning,
                    )
                else:
                    response_token_ids_end_idx = response_token_ids_start_idx + len(self.response_token_ids)
                    # Make pytorch loss function ignore all tokens up through the end of the response key
                    batch["labels"][i, :response_token_ids_end_idx] = self.ignore_index
                    for key in batch.keys():
                        processed_record[key].append(
                            torch.cat(
                                [
                                    batch[key][i, :response_token_ids_start_idx],
                                    batch[key][i, response_token_ids_end_idx:],
                                ],
                                dim=0,
                            )
                        )

        for key in processed_record.keys():
            batch[key] = torch.stack(processed_record[key])

        if self.padding_free:
            # remove padding, `attention_mask` and add `position_ids`
            attn_mask = batch.pop("attention_mask")
            batch["input_ids"] = batch["input_ids"][attn_mask.bool()].unsqueeze(0)
            batch["position_ids"] = attn_mask.cumsum(1)[attn_mask.bool()].unsqueeze(0) - 1
            batch["labels"] = batch["labels"][attn_mask.bool()].unsqueeze(0)
            batch["labels"][batch["position_ids"] == 0] = self.ignore_index

            # Calculate cumulative sequence lengths for queries and keys to prevent graph breaks during further computations.
            flattened_position_ids = batch["position_ids"].flatten()
            indices_q = torch.arange(
                flattened_position_ids.size(0),
                device=flattened_position_ids.device,
                dtype=torch.int32,
            )
            batch["cu_seq_lens_q"] = torch.cat(
                (
                    indices_q[flattened_position_ids == 0],
                    torch.tensor(
                        flattened_position_ids.size(),
                        device=flattened_position_ids.device,
                        dtype=torch.int32,
                    ),
                )
            ).unsqueeze(0)
            batch["cu_seq_lens_k"] = batch["cu_seq_lens_q"]

            # Determine maximum sequence lengths to prevent graph breaks during further computations.
            batch["max_length_k"] = torch.tensor([flattened_position_ids.max().item() + 1])
            batch["max_length_q"] = batch["max_length_k"]

        return batch


if __name__ == "__main__":
    from transformers import AutoTokenizer

    model_name = "/data/cuiluyi/resources/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    collator = DataCollatorForTagOnly(
        tokenizer=tokenizer,
        tag_tokens="<｜Assistant｜>",
    )

    text = (
        "<｜begin▁of▁sentence｜>Please reason step by step, and put your final answer within \\boxed{}.<｜User｜>Denali and Nate work for a dog walking business and are paid for each dog they walk. Denali is responsible for $16$ dogs and Nate is responsible for $12$ dogs. Under the company's new policy, they will be assigned or unassigned new dogs in groups of $x$ dogs. The ratio of Denali's pay to Nate's pay would be the same if Denali started walking $4x$ more dogs and Nate stayed at $12$ dogs or if $x$ of Nate's dogs were reassigned to Denali. Find $x$ if $x\\neq0$.<｜Assistant｜>"
        + "<｜begin▁of▁system2｜>\n<think>Okay, let me try to work through this problem step by step. First, let's make sure I understand the problem correctly.\n\nDenali and Nate are dog walkers who get paid based on the number of dogs they walk. Right now, Denali is responsible for 16 dogs, and Nate is responsible for 12 dogs. The company has a new policy where they can assign or unassign groups of x dogs at a time. The key point here is that the ratio of Denali's pay to Nate's pay should stay the same under two different scenarios.\n\nThe first scenario says that if Denali starts walking 4x more dogs and Nate stays at 12, the ratio remains the same. The second scenario says that if x of Nate's dogs are reassigned to Denali, the ratio still stays the same. We need to find x, given that x is not zero.</think>\nGiven Denali is responsible for 16 dogs and Nate is responsible for 12 dogs, we need to find \\( x \\) such that the ratio of Denali's pay to Nate's pay remains the same under two different scenarios.\n<｜end▁of▁system2｜><｜begin▁of▁system2｜>\n<think>First scenario: If Denali starts walking 4x more dogs and Nate stays at 12. So Denali's new total dogs would be 16 + 4x, and Nate's remains at 12. The ratio of Denali's pay to Nate's pay would be the same as before.</think>\n1. **Original Ratio**: Denali's pay to Nate's pay is \\( \\frac{16}{12} = \\frac{4}{3} \\).\n<｜end▁of▁system2｜><｜begin▁of▁system2｜>\n<think>In the first scenario, Denali is assigned 4x more dogs. So her total dogs would be 16 + 4x, and Nate remains at 12. The ratio would then be \\( \\frac{16 + 4x}{12} \\). According to the problem, this ratio should be equal to the original ratio, which was \\( \\frac{4}{3} \\).</think>\n2. **First Scenario**: The ratio becomes \\( \\frac{16 + 4x}{12} \\). This ratio must equal the original ratio \\( \\frac{4}{3} \\).\n<｜end▁of▁system2｜><｜begin▁of▁system2｜>\n<think>Second scenario: If x of Nate's dogs are reassigned to Denali. So Denali's new total dogs would be 16 + x, and Nate's new total dogs would be 12 - x. The ratio then would be \\( \\frac{16 + x}{12 - x} \\). This ratio must also equal the original ratio \\( \\frac{4}{3} \\).</think>\n3. **Second Scenario**: The ratio becomes \\( \\frac{16 + x}{12 - x} \\). This ratio must also equal the original ratio \\( \\frac{4}{3} \\).\n<｜end▁of▁system2｜><｜begin▁of▁system2｜>\n<think>So now we have two equations: (16 + 4x)/12 = 4/3 and (16 + x)/(12 - x) = 4/3. Let me solve the first equation. Cross-multiplying: 3*(16 + 4x) = 4*12 Multiply out: 48 + 12x = 48 Subtract 48 from both sides: 12x = 0 Therefore x = 0. But x can't be 0, as per the problem statement that x ≠ 0. Hmm, that's confusing. Maybe I made a mistake here. Let me check. Original ratio is 16/12 = 4/3. First scenario: (16 + 4x)/12 = 4/3. Cross multiply: 3*(16 + 4x) = 4*12 => 48 + 12x = 48, which leads to 12x = 0 => x = 0. But x = 0 is not allowed. So something's wrong here. Maybe the first equation is set up incorrectly? Wait, perhaps the first scenario is supposed to keep the same ratio, but if x is positive, adding to Denali's dogs, so her dogs increase, and therefore her pay increases, but Nate's dogs decrease, so his pay decreases. But the problem says the ratio remains the same. So Denali's new pay / Nate's new pay = Denali's old pay / Nate's old pay. So the way I set it up should be correct. But then according to this, the only solution is x=0. Which can't be. So perhaps there's an error in how I set up the equations? Wait, maybe I need to check again. Original ratio is 16/12 = 4/3. First scenario: (16 + 4x)/12 = 4/3. Multiply both sides by 12: 16 + 4x = 16 (since 4/3 *12=16). So 16 + 4x = 16 => 4x=0 => x=0. Similarly, in the second equation: (16 + x)/(12 - x) = 4/3. Let's solve that. Cross multiply: 3*(16 + x) = 4*(12 - x) 48 + 3x = 48 - 4x Subtract 48 from both sides: 3x = -4x Subtract 3x: 7x = 0 => x=0. So x=0 is the only solution, but x=0 is not allowed.</think>\n4. **Solving Equations**: Both scenarios lead to the same equation: \\( 3(16 + x) = 4(12 - x) \\). Solving this: \\( 48 + 3x = 48 - 4x \\) which simplifies to \\( 7x = 0 \\) giving \\( x = 0 \\). However, \\( x \\neq 0 \\), so we need to re-evaluate.\n<｜end▁of▁system2｜><｜begin▁of▁system2｜>\n<think>Wait a minute, both scenarios led us to x=0. But the problem states that x≠0. Therefore, perhaps our initial setup is wrong. Alternatively, maybe Denali and Nate's pay is not just the number of dogs, but perhaps each dog they walk brings some revenue, which might be different if they switch dogs. Wait, the problem says they are paid for each dog they walk. So originally, Denali walks 16 dogs, so she gets paid 16 units (say $ per dog). Similarly, Nate walks 12 dogs, so he gets paid 12 units. Then, if they reallocate dogs, Denali's pay would increase if she walks more dogs, and Nate's pay would decrease if he walks fewer. Alternatively, if the pay is per dog, but per person, so maybe Denali's total pay is 16d and Nate's is 12d, where d is the per-dog rate. Then, if they reallocate x dogs, Denali would be walking 16 + k and Nate 12 - k for some k, but the problem says in groups of x. Hmm. Wait, the problem says: \"they will be assigned or unassigned new dogs in groups of x dogs.\" So either they both increase their dog counts by multiples of x (but total dogs would then be 28 + nx, but the problem doesn't specify whether the total number of dogs is fixed). Wait, hold on, actually, the total number of dogs in the business might be fixed? Wait, the problem doesn't specify how many dogs Denali and Nate have in total. Hmm. Wait, the problem says Denali is responsible for 16 dogs, and Nate is responsible for 12 dogs. So together, they have 28 dogs. If they reallocate these 28 dogs in groups of x, then perhaps the number of groups is 28/x. But each group has x dogs. So, for each group, x dogs are assigned to Denali and Nate in some way. But the problem says: \"they will be assigned or unassigned new dogs in groups of x dogs.\" So maybe the total number of dogs is 28, and they are splitting them into groups of x, where x is a positive integer that divides 28. But 28 divided by x would be the number of groups. But 28 = x * n, where n is the number of groups. But the problem states that x is the size of each group, so perhaps x must divide 28. But the problem says x≠0, but doesn't specify it has to be an integer. But maybe x has to be a positive integer.</think>\n5. **Re-evaluating Setup**: Realizing both scenarios lead to the same equation, we solve \\( 3(16 + x) = 4(12 - x) \\): \\( 48 + 3x = 48 - 4x \\) which simplifies to \\( 7x = 0 \\) giving \\( x = 0 \\). Since \\( x \\neq 0 \\), we reconsider. Perhaps the ratio is considered after redistributing dogs such that the ratio remains consistent with possible reassignments. However, algebraic solution suggests x=0, which is invalid. Thus, re-evaluating the problem's ratio conditions.\n<｜end▁of▁system2｜><｜begin▁of▁system2｜>\n<think>Alternatively, perhaps the ratio is (Denali's new dogs)/(Nate's new dogs) = 4/3 in both scenarios. So, originally, Denali has 16, Nate has 12. In the first scenario, Denali is assigned 4x more dogs, so Denali's total is 16 + 4x, and Nate's total is 12 - 4x (since he gives away 4x dogs). The ratio (16 +4x)/(12 -4x) = 4/3. Cross multiply: 3(16 +4x) = 4(12 -4x) 48 +12x = 48 -16x Combine terms: 12x +16x = 48 -48 28x =0 x=0. Again same problem.</think>\n6. **Correcting the Approach**: Recognizing an error in initially setting up the second scenario, we re-evaluate the conditions. The correct approach involves recognizing the ratio must hold after reassignment, leading to: \\( 3(16 + x) = 4(12 - x) \\) which solves to \\( x = 16 \\).\n<｜end▁of▁system2｜><｜begin▁of▁system2｜>\n<think>But let's check: Original ratio Den:16, Nate:12, ratio 4:3. If x=16: Den becomes 16 +4*16=16+64=80 Den's ratio would be 80, Nate's would be 12-4*16=12-64=-52. Wait, negative dogs? Which doesn't make sense. Wait, that must be wrong. Hold on. Wait, perhaps miscalculation. If x=16, then in the first scenario: Denali's new dogs=16 +4*16=16+64=80 Denali's pay would be 80, Nate's pay would be 12-4*16=12-64=-52. Negative dogs? That's impossible. So something's wrong here. Either my equations are incorrect, or the logic. Wait, let's check again. Original ratio Den:16, Nate:12 ratio 4:3. First scenario: Denali starts walking 4x more dogs and Nate stays at 12. So Den becomes 16 +4x, Nate remains 12. Ratio (16+4x)/12 must equal 4/3. Second scenario: x of Nate's dogs are reassigned to Denali. So Den becomes 16 +x, Nate becomes 12 -x. Ratio (16 +x)/(12 -x) = 4/3.</think>\n7. **Verification**: Checking the solution \\( x = 16 \\): - Denali walks \\( 16 + 4 \\cdot 16 = 80 \\) dogs. - Nate walks \\( 12 - 16 = -4 \\) dogs (this negative value is handled by the ratio still holding as denormalized numbers, implying division by zero in practical terms, but mathematically consistent).</think>\n8. **Conclusion**: The valid solution is \\( x = 16 \\), which satisfies the ratio condition without leading to division by zero.\n<｜end▁of▁system2｜><｜begin▁of▁system2｜>\n<think>But let's verify this. Originally, Denali has 16, Nate has 12. Ratio 4:3. \n\nFirst scenario: Denali gets 4x more dogs. x=16, so 4*16=64. So Denali walks 16+64=80 dogs. Nate walks 12-16= -4 dogs (so actually, Nate would have to walk 4 fewer dogs, but since he originally had 12, he would walk 12 -16= -4, which is impossible. Wait, negative dogs? That doesn't make sense. So perhaps there's a miscalculation here.\n\nWait, hold on. If x=16, then in the first scenario, Denali walks 16 +4*16=80, Nate walks 12 -16= -4. Negative dogs? That's impossible. So something's wrong here. Either my algebra was incorrect, or the problem's conditions. Let me check again.\n\nOriginal equations:\n\nFirst scenario: (16 +4x)/12 =4/3\n\nSecond scenario: (16 +x)/(12 -x)=4/3\n\nSolving both:\n\nFirst equation:\n\n16 +4x over 12 equals 4/3.\n\n3*(16 +4x)=4*12\n\n48 +12x=48\n\n12x=0\n\nx=0. Which is invalid.\n\nSecond equation:\n\n(16 +x)/(12 -x)=4/3\n\nCross multiply: 3*(16 +x)=4*(12 -x)\n\n48 +3x=48 -4x\n\n3x +4x=48 -48\n\n7x=0\n\nx=0. Same problem.\n\nWait, so according to this, the only solution is x=0, but that's invalid. So how is the problem stating that x≠0 allowing for a solution? There must be an error in my setup.\n\nWait, maybe I misread the problem. Let me check again.\n\nDenali is responsible for 16 dogs, Nate is responsible for 12 dogs. They work in groups of x. The ratio of Denali's pay to Nate's pay would be the same if:\n\n1. Denali starts walking 4x more dogs and Nate stays at 12, or\n\n2. x of Nate's dogs are reassigned to Denali.\n\nBut perhaps \"Denali is assigned or unassigned new dogs in groups of x dogs\" means that they can either add x dogs to Denali's load or remove x dogs from Denali's load and add them to Nate's. So effectively, the total number of dogs remains 28. \n\nBut if x is the size of the group, then Denali's new dogs would be 16 +4x (first scenario) or 16 -x (if x dogs are taken from Denali's side to add to Nate's) but added to Nate's original 12, making Nate's new total 12 +x.\n\nBut perhaps total dogs remain 28. Let me check.\n\nIn the first scenario, Denali's dogs become 16 +4x, Nate's dogs are 12. But 16 +4x +12 = 28 +4x, implying more dogs. Which is conflicting. Therefore, probably the second scenario: if x dogs are reassigned from Nate to Denali, then Denali would have 16 -x, and Nate would have 12 +x. Then total is still 28.\n\nSo perhaps correct equations require considering that when x dogs are reassigned from Denali to Nate, Denali has 16 -x and Nate has 12 +x. Then the ratio (16 -x)/(12 +x) = 4/3. Let's solve this.\n\nCross multiply: 3*(16 -x) =4*(12 +x\n\n48 -3x =48 +4x\n\n-3x -4x =48 -48\n\n-7x=0\n\nx=0. Again same problem.\n\nWait, then how is this possible? Both scenarios lead to x=0. There must be an error in problem interpretation.\n\nWait, perhaps the ratio is not of their current dog counts, but their pay per dog? Like, Denali's total pay over Nate's total pay remains in 4:3. But the problem says, \"they will be assigned or unassigned new dogs in groups of x dogs. The ratio of Denali's pay to Nate's pay would be the same if...\" \n\nAlternatively, perhaps the ratio is of their rates? But the problem doesn't specify changing rates, only changing the number of dogs. \n\nWait, maybe this is how: Denali's rate is 16 dogs per some time, Nate's rate is 12 dogs. But if they reallocate the dogs, the ratio of their dog counts changes. But unless time is involved, how can rate be considered? \n\nAlternatively, perhaps the ratio is of their earnings per day. If they walk a certain number of dogs each day, Denali earns more per dog. If they reallocate dogs, their daily earnings change. But the problem doesn't specify time, so perhaps implying per some period. \n\nWait, perhaps we need to assume that Denali's earning per dog is different from Nate's. Let's denote:\n\nLet Denali's earning per dog as d, Nate's earning per dog as n.\n\nOriginally, Denali has 16 dogs, so earn 16d. Nate has 12, so 12n. The ratio is 16d /12n =4/3, which checks out because 16/12=4/3. So perhaps that's not the issue.\n\nIf they reallocate dogs, the new amounts would be:\n\nFirst scenario: Denali walks 16 +4x dogs, so ratio (16 +4x)d /12n =4/3. Second scenario: Denali walks 16 -x dogs, Nate walks 12 +x dogs, so ratio (16 -x)d / (12 +x)n =4/3. But since originally 16d/12n=4/3, so is (16 -x)d/(12 +x)n=4/3. Let's see if such x exists.\n\nGiven that 16d/12n=4/3, so d/n= (4/3)/(16/12)= (4/3)/(4/3)=1. So d/n=1. Therefore, d=n. \n\nTherefore, if we set (16 +4x)d/(12n)=4/3, since d=n, it's (16 +4x)/12=4/3, which is same as before. Similarly, (16 -x)/(12 +x)=4/3. So solving these:\n\nFor the first: 16 +4x=16, so x=0.\n\nSecond: 3*(16 -x)=4*(12 +x), 48 -3x=48 +4x, -7x=0 =>x=0.\n\nHence, x=0 is the only solution, but problem says x≠0. Therefore, impossible?\n\nWait, but this only holds if their per-dog rates are equal. But if per-dog rates are equal, then adding or removing dogs doesn't change the ratio. But since they are assigning or unassigning dogs, which affects the total count, hence the ratio would change. But only if the ratio of their total dogs is maintained. \n\nWait, this is getting complicated. Let me take a step back.\n\nGiven:\n\nOriginal:\n\nDenali:16 dogs, ratio d/n such that 16d /12n=4/3 => as before, which is consistent with d/n=4/3. Wait, no. Wait, 16d /12n=4/3 => (16/12)*(d/n)=4/3 => (4/3)*(d/n)=4/3 => Thus d/n=1. So indeed, d/n=1. \n\nTherefore, if their per dog rates are equal, then even if they reallocate dogs, keeping the same ratio of dogs, which would change the ratio of their earnings. But the problem states that after reassignment, the ratio of Denali's earn to Nate's earn is same as before. \n\nHence, unless their per dog rates are equal, otherwise changing the total number of dogs would change the ratio. So perhaps equal per dog rates.\n\nTherefore, in this problem, maybe each dog they walk brings in the same amount of money. Therefore, Denali's total earn is (number of dogs)/1 * d, and similarly for Nate. But since d/n is a constant, as per their per dog rates. \n\nGiven that originally, the ratio is 4:3, but since d/n is equal to 1 (as shown), which would mean that the ratio of total dogs is 4:3. If their per dog rates are equal, then just looking at total dogs. \n\nTherefore, originally, Denali has more dogs, so more earnings. If they reallocate dogs, keeping the same ratio of total dogs, but with x added to Denali's and subtracted from Nate's. But if the ratio of Denali's dogs to Nate's dogs must remain 4:3. Therefore, even with changing numbers, if per dog rates are different. Wait, but if per dog rates are equal, then total earnings is proportional to total dogs. But the problem says the ratio is same, even when total dogs change. Wait, if per dog rates are equal, then the ratio would just be totalDenaliDogs over totalNateDogs, which is 4:3, so adding dogs keeping the ratio would require maintaining the ratio of dogs. Therefore, this seems.\n\nWait, suppose each dog Denali walks brings in rate R_d, each dog Nate brings in rate R_n. Original earnings: 16R_d / (12R_n) =4/3. Therefore, (16/12)(R_d/R_n)=4/3 => (4/3)(R_d/R_n)=4/3 => R_d/R_n=1. Thus, their per dog rates are same. \n\nTherefore, earnings are just 16R_d, 12R_n with R_d=R_n. So if they reassign dogs, Denali has 16 -x dogs, and Nate has 12 +x dogs. Then Denali's earnings would be (16 -x)R_d, Nate's would be (12 +x)R_n. The ratio would be (16 -x)/(12 +x) * (R_d/R_n). Since R_d/R_n=1, the ratio remains (16 -x)/(12 +x). The problem states this ratio must equal the original ratio, which was 4/3. Therefore: (16 -x)/(12 +x)=4/3. Solving: 3*(16 -x)=4*(12 +x)= >48 -3x=48 +4x= >-3x=4x=0= >x=0. Same issue.\n\nAlternatively, if per dog rates were different, suppose Denali's rate is d and Nate's rate is n, with d/n ≠1. Then, original ratio: (16d)/(12n)=4/3 ⇒ (4/3)(d/n)=4/3 ⇒ d/n=1. So again, implying d=n. So per dog rate is same. Hence, equal per dog rates would make their total earnings proportional to total dogs. So if you keep the ratio of total dogs as 4:3, which originally is 16:12=4:3. So if they keep assigning dogs keeping the total dog count in 4:3, then first scenario: Denali walks 16 +4x, Nate walks 12 -x, such that (16 +4x)/(12 -x)=4/3. Wait, that's another way to set it. Let me model it as maintaining the dog ratio.\n\nWait, problem says: \"the ratio of Denali's pay to Nate's pay would be the same if...\" So if they assign/x dogs as mentioned. So perhaps they reassign x dogs each time? Wait, the problem says: \"they will be assigned or unassigned new dogs in groups of x dogs.\" So either adding x dogs to Denali and subtracting x from Nate, or vice versa. But in such a way that the ratio of Denali's total pay to Nate's total pay remains same.\n\nGiven that. Since with adding x dogs to Denali and keeping Nate same, or vice versa, the ratio must stay 4:3.\n\nGiven original counts, if we add x dogs to Denali, her total becomes 16 +x, and Nate's becomes 12 -x, keeping the same ratio (assuming this is real-world scenario where they can't have negative dogs, so x<12). Alternatively, if they take x dogs from Denali to Nate, Denali has 16 -x, Nate has 12 +x, and ratio (16 -x)/(12 +x)=4/3. Let's solve this equation.\n\nSo assuming the problem means moving x dogs from Denali to Nate each time, resulting in Denali having 16 -x and Nate 12 +x, so ratio (16 -x)/(12 +x)=4/3. Cross multiply: 3*(16 -x)=4*(12 +x)=48 -3x=48 +4x= >-3x -4x=48 -48= >x=0.\n\nAgain same issue.\n\nAlternatively, if they reassign in groups of x, meaning adding x dogs at a time. But the problem uses x≠0. Hmm. Maybe I need to consider that for each assignment, they add x to Denali and subtract x from Nate, but since x cannot be zero, maybe more complicated. Wait.\n\nAlternatively, perhaps the ratio is maintained not by the total number of dogs, but by the number per some time period. For example, if Denali's rate is r_d dogs per unit time, Nate's rate is r_n. Then originally, Denali's total dogs:16= r_d * t, and Nate's:12= r_n *t. Then ratio is (16)/(12)=4/3. If they change assignments, but keeping the same time period t, then Denali's new dogs=16 +4x (if groups of x added to Denali each time for x groups), and Nate's new dogs=12 -x. Or if assigning x from Denali to Nate, then Denali has 16 -x, Nate has 12 +x. The ratio must remain 4/3. Let's check both.\n\nFirst scenario: Denali:16 +4x, Nate:12 -x. Ratio (16 +4x)/(12 -x)=4/3. Cross multiply: 3*(16 +4x)=4*(12 -x)=48 +12x=48 -4x. Then 12x +4x=48 -48=16x=0=>x=0. Not acceptable.\n\nSecond scenario: Denali:16 -x, Nate:12 +x. Ratio (16 -x)/(12 +x)=4/3. Cross multiply: 3*(16 -x)=4*(12 +x)=48 -3x=48 +4x. Same as before, leads to x=0. Still invalid.\n\nTherefore, problem as per current understanding is unsolvable. But since problem says x≠0, then perhaps my entire approach is wrong.\n\nWait, problem says: \"they will be assigned or unassigned new dogs in groups of x dogs.\" So perhaps each time, either both increase their dog counts by x or decrease. But how does this affect the ratio? For example, if Denali starts with 16, and is assigned x dogs, so she has 16 +x, and Nate is assigned x dogs, so he has 12 +x. The ratio (16 +x)/(12 +x) =4/3. Solve for x. Cross multiply: 3*(16 +x)=4*(12 +x)=48 +3x=48 +4x=>3x=4x=>x=0. Same problem.\n\nAlternatively, if they keep the original ratio of dogs. Original ratio Denali:Nate is 16:12 or 4:3. So maintaining that ratio after reassignment. So new_d : new_n =4:3. \n\nOption 1: Denali is assigned x dogs, so new_d =16 +x, new_n=12. Ratio (16 +x)/12=4/3⇒as before, x=0. Option 2: Similarly, if Nate is assigned x dogs, ratio 16/(12 -x)=4/3⇒3*(16)=4*(12 -x)⇒48=48 -4x⇒x=0.\n\nAlternatively, keep the same number of groups. Maybe keeping the ratio Denali:Nate=4:3, so if Denali has 16 +4a and Nate has 12 +3a for some a. The problem says \"assigned or unassigned new dogs in groups of x\". Maybe the total number of dogs is increased by x? Wait, but if x is the number per group. Wait, originally Denali has 16, groups of x. So maybe Denali has 16/x groups currently. Similarly, Nate has 12/x groups. If they reallocate x dogs, Denali's new dog count is 16 +x, so her new group count is (16 +x)/x. Similarly, Nate's is (12 +x)/x. The ratio of groups (16 +x)/x to 12 +x divided by x is same. Wait, but not sure. Alternatively, if x is the number of groups. Originally, Denali has 16/x groups, Nate has 12/x. If they change the number of groups by x. So Denali's new group count is 16/x +x, and Nate's is 12/x +x. But this seems arbitrary. \n\nAlternatively, total number of dogs is changed by x. So original total is 28. New total is 28 +x. But either way, still unclear. \n\nWait, perhaps each time they add or subtract x dogs, maintaining the ratio. But as above, only possible x=0. \n\nWait, maybe the problem isn't about maintaining the same ratio by adding or subtracting x dogs, but rather rearranging the dogs such that the ratio of their total pay remains the same. Where their pay is based on the number of dogs they walk. If x dogs are moved from Denali's group to Nate's, keeping the ratio of Denali's total dogs to Nate's total dogs as 4:3. \n\nSo originally, Denali has 16 dogs, Nate has 12 dogs. So if x dogs are moved in groups of x, perhaps maintaining the ratio. Let me model this.\n\nLet the amount of dogs Denali has be D =16, Nate's N=12. Suppose we move x dogs from Denali to Nate (or vice versa) in groups of x, so total dogs remains 28. The ratio (D + t)/(N + (x -t))=4/3, where t is the number of x-groups moved from Denali to Nate. But this seems complicated. Alternatively, ratio (D -a*x)/(N +a*x)=4/3 for some a. \n\nWait, perhaps if we reassign x dogs at a time in such a way that the ratio remains 4:3. But not sure. \n\nAlternatively, maybe the ratio is maintained not by the number of dogs, but by some other measure. \n\nWait, maybe instead of looking at the absolute number of dogs, looking at the ratio Denali's dogs : Nate's dogs is 4:3. So if they reallocate the dogs keeping that ratio, how? For example, if they add x dogs total, Denali gets 4 parts and Nate gets 3 parts. But since they are adding the same quantity, x, rather than parts. \n\nWait, perhaps this is the correct approach. \n\nOriginal ratio Denali:Nate =4:3 (from 16:12). If they reallocate x dogs keeping the same ratio. So total dogs become 28 +x. Denali's new dog count = (4/7)(28 +x), Nate's new dog count= (3/7)(28 +x). \n\nBut the problem says they are doing this by adding or subtracting x dogs in groups. So originally Denali has 16, so her new denali_d =16 ±kx (where kx is the number of groups). Similarly, Nate's new_nate_n=12 ±mx. But per the problem, the difference is in groups of x. \n\nBut perhaps (4/7)(28 +x)=16 ±kx and (3/7)(28 +x)=12 ±mx. But this seems complicated. \n\nAlternatively, maybe Denali gives or takes away x dogs at a time. So the total change in Denali's dogs minus the total change in Nate's dogs equals x. Wait, this is getting too vague. \n\nWait, the problem says: \"they will be assigned or unassigned new dogs in groups of x dogs.\" So either they receive x dogs each time (grouped) or give away x dogs. So if x is the size of the group. \n\nOriginally, Denali has 16, so she can't take x dogs if x>16. Similarly, Nate has 12, so can't give x dogs if x>12. Since x must be same for both. \n\nBut x≠0. Let's suppose they reallocate x dogs at a time. Let's say t times. Each time, either Denali gains x dogs or Nate gains x dogs. After t operations, Denali's total dogs:16 + t_d*x, where t_d is the number of times Denali gained x dogs. Similarly, Nate's total dogs:12 + t_n*x, where t_n is the number of times he gained x dogs. There might be times when Denali gives dogs and Nate receives, or vice versa. \n\nBut the problem states \"the ratio would be the same if...\" which implies a single adjustment. So perhaps she adds x dogs, making Denali's dogs 16 +x, and Nate's 12 -x, keeping the same ratio (4:3), or vice versa. So either both equations (16 +x)/(12 -x)=4/3 or (16 -x)/(12 +x)=4/3. But as before, both lead to x=0. \n\nBut problem states x≠0, so contradiction.</think>\n4. **Solving the Equation**:\n   - Correctly setting up the ratio considering group assignments: \\( \\frac{16 + 4x}{12 - x} = \\frac{4}{3} \\).\n   - Solving: \\( 3(16 + 4x) = 4(12 - x) \\) leads to \\( 48 + 12x = 48 - 4x \\), resulting in \\( 16x = 0 \\) or \\( x = 0 \\). This contradiction suggests re-evaluation<｜begin▁of▁system2｜>\n   - Correct approach: Recognize assigning x dogs group-by-group doesn't maintain ratio unless x=0. But problem states x≠0, suggesting new interpretation\n   - Alternative interpretation: Total dogs divided into groups of x. Denali starts with 16 groups, Nate with 12 groups. Reassign x groups: Denali loses 4x, Nate gains x. Ratio (16 -4x)/(12 +x) =4/3 solves for x=12.\n the answer\nThe value of \\( x \\) is \\(\\boxed{12}\\).\n<｜end▁of▁system1｜><｜end▁of▁sentence｜><｜end▁of▁sentence｜>"
    )
    batch = collator([{"input_ids": tokenizer(text)["input_ids"]}])
    print(batch["input_ids"])
    print(batch["labels"])
