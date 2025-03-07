import paramiko
import os
from tqdm import tqdm
import argparse
from servers_info import servers
# crontab -e 来添加定时任务
# */5 * * * * /usr/bin/python3 /home/lkx/Documents/gym/LocomotionWithNP3O/servers_tools/file_sync.py
def count_files(sftp, remote_path):
    """
    递归统计远程目录中的文件数量
    """
    file_count = 0
    for item in sftp.listdir_attr(remote_path):
        item_remote_path = os.path.join(remote_path, item.filename)
        if item.st_mode & 0o40000:  # 如果是文件夹
            file_count += count_files(sftp, item_remote_path)
        else:  # 如果是文件
            file_count += 1
    return file_count

def count_need_sync_files(sftp, remote_path, local_path):
    """
    递归统计需要同步的文件数量
    """
    file_count = 0
    for item in sftp.listdir_attr(remote_path):
        item_remote_path = os.path.join(remote_path, item.filename)
        item_local_path = os.path.join(local_path, item.filename)

        if item.st_mode & 0o40000:  # 如果是文件夹
            file_count += count_need_sync_files(sftp, item_remote_path, item_local_path)
        else:  # 如果是文件
            if os.path.exists(item_local_path):
                local_mtime = os.path.getmtime(item_local_path)
                local_size = os.path.getsize(item_local_path)
                if item.st_mtime <= local_mtime and item.st_size == local_size:
                    continue
            file_count += 1
    return file_count

def sync_remote_folder(host, port, username, password, remote_folder, local_folder):
    # 创建 SSH 客户端
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        # 连接到远程服务器
        ssh.connect(host, port=port, username=username, password=password)
        sftp = ssh.open_sftp()

        # 统计需要同步的文件总数
        total_files = count_need_sync_files(sftp, remote_folder, local_folder)

        def recursive_sync(remote_path, local_path, pbar):
            # 确保本地目录存在
            if not os.path.exists(local_path):
                os.makedirs(local_path)

            # 获取远程目录中的所有文件和文件夹
            for item in sftp.listdir_attr(remote_path):
                item_remote_path = os.path.join(remote_path, item.filename)
                item_local_path = os.path.join(local_path, item.filename)

                if item.st_mode & 0o40000:  # 如果是文件夹
                    recursive_sync(item_remote_path, item_local_path, pbar)
                else:  # 如果是文件
                    if os.path.exists(item_local_path):
                        local_mtime = os.path.getmtime(item_local_path)
                        local_size = os.path.getsize(item_local_path)
                        if item.st_mtime <= local_mtime and item.st_size == local_size:
                            continue
                    pbar.set_description(f"正在同步文件: {item_remote_path}")
                    sftp.get(item_remote_path, item_local_path)
                    pbar.update(1)

        # 创建进度条
        with tqdm(total=total_files, desc="同步进度") as pbar:
            # 开始递归同步
            recursive_sync(remote_folder, local_folder, pbar)

        # 关闭 SFTP 连接和 SSH 连接
        sftp.close()
        ssh.close()
        print("同步完成")
    except Exception as e:
        print(f"同步失败: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="同步指定主机名称的文件")
    # 修改 default 参数为列表形式
    parser.add_argument("--name", help="要同步的主机名称列表，多个名称用空格分隔", default="rl_env2")
    args = parser.parse_args()

    name = args.name
    for server in servers:
        if server["name"] == name:
            print(f"开始同步 {name} ({server['host']}) 的文件...")
            sync_remote_folder(
                server["host"],
                server["port"],
                server["username"],
                server["password"],
                server["remote_folder"],
                server["local_folder"]
            )
            break
    else:
        print(f"未找到名称为 {name} 的主机配置信息。")
