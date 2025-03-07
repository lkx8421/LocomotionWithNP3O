import paramiko
# 从 servers_info.py 文件中导入 servers 列表
from servers_info import servers

for server in servers:
    try:
        # 创建 SSH 对象
        ssh = paramiko.SSHClient()
        # 允许连接不在 know_hosts 文件中的主机
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        # 连接服务器
        ssh.connect(hostname=server['host'], port=server['port'], username=server['username'], password=server['password'])
        # 执行命令
        stdin, stdout, stderr = ssh.exec_command('nvidia-smi')
        # 获取命令结果
        result = stdout.read().decode()
        # 关闭文件对象
        stdin.close()
        stdout.close()
        stderr.close()
        # 打印结果
        print(f"Server: {server['host']}")
        print(result)
        # 关闭连接
        ssh.close()
    except Exception as e:
        print(f"Error connecting to {server['host']}: {e}")