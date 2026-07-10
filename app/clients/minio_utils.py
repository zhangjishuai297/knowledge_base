# 导入Python内置模块
import os
import json
from minio import Minio
# 项目内部配置与日志
from app.conf.minio_config import minio_config
from app.core.logger import logger

# 全局变量，先置空，懒加载
minio_client = None
bucket_name = minio_config.bucket_name

def get_minio_client():
    """懒加载获取MinIO单例，仅首次调用初始化连接"""
    global minio_client
    if minio_client is not None:
        return minio_client
    
    try:
        # 仅第一次调用才创建客户端
        minio_client = Minio(
            endpoint=minio_config.endpoint,
            access_key=minio_config.access_key,
            secret_key=minio_config.secret_key,
            secure=minio_config.minio_secure
        )
        logger.info("[MinIO客户端] 客户端实例初始化成功")

        # 桶不存在则创建 + 仅创建时配置策略，不重复执行
        if not minio_client.bucket_exists(bucket_name):
            logger.info(f"[MinIO客户端] 存储桶 '{bucket_name}' 不存在，正在创建...")
            minio_client.make_bucket(bucket_name)
            logger.info(f"[MinIO客户端] 存储桶 '{bucket_name}' 创建成功")

            # 仅新建桶时配置公开策略，避免重复请求
            policy = {
                "Version": "2012-10-17",
                "Statement": [{
                    "Effect": "Allow",
                    "Principal": {"AWS": ["*"]},
                    "Action": ["s3:GetObject"],
                    "Resource": [f"arn:aws:s3:::{bucket_name}/*"]
                }]
            }
            minio_client.set_bucket_policy(bucket_name=bucket_name, policy=json.dumps(policy))
            logger.info(f"[MinIO客户端] 存储桶 '{bucket_name}' 公网只读策略已配置成功")
        else:
            logger.info(f"[MinIO客户端] 存储桶 '{bucket_name}' 已存在，无需重复创建")
        
        return minio_client
    except Exception as e:
        logger.error(f"[MinIO客户端] 初始化失败: {e}")
        minio_client = None
        raise e

# 测试入口，仅直接运行文件时执行
if __name__ == "__main__":
    client = get_minio_client()
    bucket_policy = client.get_bucket_policy(bucket_name)
    logger.info(f"[MinIO客户端] 存储桶 '{bucket_name}' 当前策略：{bucket_policy}")