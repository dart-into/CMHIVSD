import subprocess
import sys

# 定义命令参数（建议拆分为列表，避免路径空格问题）
command = [
    sys.executable,  # 使用当前Python解释器路径
    "detect.py",
    "--source", "split",
    "--weights", "runs/train/exp29/weights/best.pt",
    "--save-txt"
]

# 执行命令并捕获输出
result = subprocess.run(
    command,
    capture_output=True,  # 捕获标准输出和错误
    text=True             # 以文本形式返回结果
)

# 输出结果
print("标准输出:", result.stdout)
print("标准错误:", result.stderr)
print("返回码:", result.returncode)

# 检查是否执行成功
if result.returncode == 0:
    print("检测脚本执行成功！")
else:
    print("检测脚本执行失败！")