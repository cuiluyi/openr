import yaml
 
data = {
    "Person": {
        "name": "John",
        "age": 30,
        "address": {
            "street": "123 Main St",
            "city": "Anytown",
            "state": "CA"
        }
    }
}
# 将 data 变量存储的数据写入 YAML 文件
with open(file="example.yaml", mode="w") as f:
    yaml.dump(data, f)