# ERROR

## requests.exceptions.ConnectionError

### 问题描述

```
Traceback (most recent call last):
  File "/data/cuiluyi/openr/reason/evaluation/evaluate.py", line 254, in <module>
    parallel_evaluate_test_dataset(config.method, solver_fn, save_dir)
  File "/data/cuiluyi/openr/reason/evaluation/evaluate.py", line 164, in parallel_evaluate_test_dataset
    for i, item in enumerate(
  File "/data/cuiluyi/anaconda3/envs/open_reasoner/lib/python3.10/site-packages/tqdm/std.py", line 1181, in __iter__
    for obj in iterable:
  File "/data/cuiluyi/anaconda3/envs/open_reasoner/lib/python3.10/site-packages/ray/util/actor_pool.py", line 170, in get_generator
    yield self.get_next_unordered()
  File "/data/cuiluyi/anaconda3/envs/open_reasoner/lib/python3.10/site-packages/ray/util/actor_pool.py", line 370, in get_next_unordered
    return ray.get(future)
  File "/data/cuiluyi/anaconda3/envs/open_reasoner/lib/python3.10/site-packages/ray/_private/auto_init_hook.py", line 21, in auto_init_wrapper
    return fn(*args, **kwargs)
  File "/data/cuiluyi/anaconda3/envs/open_reasoner/lib/python3.10/site-packages/ray/_private/client_mode_hook.py", line 103, in wrapper
    return func(*args, **kwargs)
  File "/data/cuiluyi/anaconda3/envs/open_reasoner/lib/python3.10/site-packages/ray/_private/worker.py", line 2745, in get
    values, debugger_breakpoint = worker.get_objects(object_refs, timeout=timeout)
  File "/data/cuiluyi/anaconda3/envs/open_reasoner/lib/python3.10/site-packages/ray/_private/worker.py", line 901, in get_objects
    raise value.as_instanceof_cause()
ray.exceptions.RayTaskError(ConnectionError): ray::RemoteMathEvaluator.evaluate_problem() (pid=3273031, ip=10.208.41.156, actor_id=4b2cdbacf696a88ad97b930f01000000, repr=<reason.evaluation.evaluator.RemoteMathEvaluator object at 0x7f7f03b07df0>)
  File "/data/cuiluyi/anaconda3/envs/open_reasoner/lib/python3.10/site-packages/urllib3/connectionpool.py", line 536, in _make_request
    response = conn.getresponse()
  File "/data/cuiluyi/anaconda3/envs/open_reasoner/lib/python3.10/site-packages/urllib3/connection.py", line 507, in getresponse
    httplib_response = super().getresponse()
  File "/data/cuiluyi/anaconda3/envs/open_reasoner/lib/python3.10/http/client.py", line 1375, in getresponse
    response.begin()
  File "/data/cuiluyi/anaconda3/envs/open_reasoner/lib/python3.10/http/client.py", line 318, in begin
    version, status, reason = self._read_status()
  File "/data/cuiluyi/anaconda3/envs/open_reasoner/lib/python3.10/http/client.py", line 287, in _read_status
    raise RemoteDisconnected("Remote end closed connection without"
http.client.RemoteDisconnected: Remote end closed connection without response

During handling of the above exception, another exception occurred:

ray::RemoteMathEvaluator.evaluate_problem() (pid=3273031, ip=10.208.41.156, actor_id=4b2cdbacf696a88ad97b930f01000000, repr=<reason.evaluation.evaluator.RemoteMathEvaluator object at 0x7f7f03b07df0>)
  File "/data/cuiluyi/anaconda3/envs/open_reasoner/lib/python3.10/site-packages/requests/adapters.py", line 667, in send
    resp = conn.urlopen(
  File "/data/cuiluyi/anaconda3/envs/open_reasoner/lib/python3.10/site-packages/urllib3/connectionpool.py", line 843, in urlopen
    retries = retries.increment(
  File "/data/cuiluyi/anaconda3/envs/open_reasoner/lib/python3.10/site-packages/urllib3/util/retry.py", line 474, in increment
    raise reraise(type(error), error, _stacktrace)
  File "/data/cuiluyi/anaconda3/envs/open_reasoner/lib/python3.10/site-packages/urllib3/util/util.py", line 38, in reraise
    raise value.with_traceback(tb)
  File "/data/cuiluyi/anaconda3/envs/open_reasoner/lib/python3.10/site-packages/urllib3/connectionpool.py", line 789, in urlopen
    response = self._make_request(
  File "/data/cuiluyi/anaconda3/envs/open_reasoner/lib/python3.10/site-packages/urllib3/connectionpool.py", line 536, in _make_request
    response = conn.getresponse()
  File "/data/cuiluyi/anaconda3/envs/open_reasoner/lib/python3.10/site-packages/urllib3/connection.py", line 507, in getresponse
    httplib_response = super().getresponse()
  File "/data/cuiluyi/anaconda3/envs/open_reasoner/lib/python3.10/http/client.py", line 1375, in getresponse
    response.begin()
  File "/data/cuiluyi/anaconda3/envs/open_reasoner/lib/python3.10/http/client.py", line 318, in begin
    version, status, reason = self._read_status()
  File "/data/cuiluyi/anaconda3/envs/open_reasoner/lib/python3.10/http/client.py", line 287, in _read_status
    raise RemoteDisconnected("Remote end closed connection without"
urllib3.exceptions.ProtocolError: ('Connection aborted.', RemoteDisconnected('Remote end closed connection without response'))

During handling of the above exception, another exception occurred:

ray::RemoteMathEvaluator.evaluate_problem() (pid=3273031, ip=10.208.41.156, actor_id=4b2cdbacf696a88ad97b930f01000000, repr=<reason.evaluation.evaluator.RemoteMathEvaluator object at 0x7f7f03b07df0>)
  File "/data/cuiluyi/openr/reason/evaluation/evaluator.py", line 123, in evaluate_problem
    solution: SolutionOutput = solver_fn(problem_inst, self.lm_call, self.rm_call)
  File "/data/cuiluyi/openr/reason/evaluation/methods.py", line 122, in beam_search
    traj_list = search_tree.beam_search(
  File "/data/cuiluyi/openr/reason/guided_search/tree.py", line 466, in beam_search
    _, _, terminated, truncated, info = new_env.step(
  File "/data/cuiluyi/openr/envs/base_env.py", line 166, in step
    self._legal_actions, api_completion_token = self.update_legal_actions()
  File "/data/cuiluyi/openr/envs/base_env.py", line 195, in update_legal_actions
    result: ConcatedLMGenResult = self.llm_gen_fn(
  File "/data/cuiluyi/openr/reason/inference/lm_call.py", line 38, in __call__
    return _generate_fastchat(
  File "/data/cuiluyi/openr/reason/inference/text_generation.py", line 55, in _generate_fastchat
    response = requests.post(
  File "/data/cuiluyi/anaconda3/envs/open_reasoner/lib/python3.10/site-packages/requests/api.py", line 115, in post
    return request("post", url, data=data, json=json, **kwargs)
  File "/data/cuiluyi/anaconda3/envs/open_reasoner/lib/python3.10/site-packages/requests/api.py", line 59, in request
    return session.request(method=method, url=url, **kwargs)
  File "/data/cuiluyi/anaconda3/envs/open_reasoner/lib/python3.10/site-packages/requests/sessions.py", line 589, in request
    resp = self.send(prep, **send_kwargs)
  File "/data/cuiluyi/anaconda3/envs/open_reasoner/lib/python3.10/site-packages/requests/sessions.py", line 703, in send
    r = adapter.send(request, **kwargs)
  File "/data/cuiluyi/anaconda3/envs/open_reasoner/lib/python3.10/site-packages/requests/adapters.py", line 682, in send
    raise ConnectionError(err, request=request)
requests.exceptions.ConnectionError: ('Connection aborted.', RemoteDisconnected('Remote end closed connection without response'))
(RemoteMathEvaluator pid=3273073) <Response [200]> [repeated 8x across cluster]
```
### 报错原因

服务器负载或并发限制：由于代码使用了 Ray 并行执行，可能对服务器发起了大量并发请求，超过了其处理能力。

### 解决办法

减小并行线程的数量

![image-20241222140910426](https://tianchou.oss-cn-beijing.aliyuncs.com/img/202412221409495.png)