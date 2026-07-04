# 导入Python内置模块
import os
import json
# 导入MinIO官方Python SDK核心类
from minio import Minio
# 项目内部配置与日志
from app.conf.minio_config import minio_config
from app.core.logger import logger

def get_minio_client() -> Minio:
    minio_client = Minio(
        endpoint=minio_config.endpoint,
        access_key=minio_config.access_key,
        secret_key=minio_config.secret_key,
        secure=minio_config.minio_secure  # 内网/本地部署用HTTP，公网部署需改为True并配置SSL
    )
    
    bucket_name = minio_config.bucket_name
    if not minio_client.bucket_exists(bucket_name):
        logger.info(f"[MinIO客户端] 存储桶 '{bucket_name}' 不存在，正在创建...")
        minio_client.make_bucket(bucket_name)
        logger.info(f"[MinIO客户端] 存储桶 '{bucket_name}' 创建成功")
    else:
        logger.info(f"[MinIO客户端] 存储桶 '{bucket_name}' 已存在，无需重复创建")
       
    # 配置存储桶公网只读策略：允许匿名用户通过URL直接访问桶内文件
    policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            # *表示所有匿名用户（S3兼容标识）
            "Principal": {"AWS": ["*"]},
            # 仅授权文件获取/访问操作
            "Action": ["s3:GetObject"],
            "Resource": [f"arn:aws:s3:::{bucket_name}/*"]
        }]
    }
    
    minio_client.set_bucket_policy(bucket_name=bucket_name, policy=json.dumps(policy))
    logger.info(f"[MinIO客户端] 存储桶 '{bucket_name}' 公网只读策略已配置成功")

    return minio_client

try:
    minio_client = get_minio_client()
except Exception as e:
    logger.error(f"[MinIO客户端] 初始化失败: {e}")

if __name__ == "__main__":
    bucket_policy = minio_client.get_bucket_policy(minio_config.bucket_name)
    logger.info(f"[MinIO客户端] 存储桶 '{minio_config.bucket_name}' 当前策略：{bucket_policy}")
