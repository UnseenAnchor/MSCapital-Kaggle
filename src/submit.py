"""通过 kagglesdk 提交预测文件到 Kaggle 比赛"""
import os, sys, time, ssl

# Python 3.9 + OpenSSL 3.x 在 Windows 加载系统证书库会崩溃 (ASN1: NOT_ENOUGH_DATA)
# 用 certifi 的 CA bundle 替代
try:
    import certifi
    _orig_load = ssl.SSLContext.load_default_certs
    def _safe_load(self, purpose=ssl.Purpose.SERVER_AUTH):
        try:
            _orig_load(self, purpose)
        except Exception:
            pass  # Windows store 失败时忽略，后续用 cafile 加载
    ssl.SSLContext.load_default_certs = _safe_load
    os.environ.setdefault('REQUESTS_CA_BUNDLE', certifi.where())
    os.environ.setdefault('SSL_CERT_FILE', certifi.where())
    print('SSL patch: certifi', certifi.where())
except ImportError:
    pass

from kagglesdk.kaggle_client import KaggleClient
from kagglesdk.competitions.types.competition_api_service import (
    ApiStartSubmissionUploadRequest, ApiCreateSubmissionRequest,
)

COMP = 'ms-capital-real-financial-market-forecasting'

def upload_file_to_signed_url(file_path, url):
    import urllib.request, ssl
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ctx = ssl._create_unverified_context()
    with open(file_path, 'rb') as f:
        data = f.read()
    req = urllib.request.Request(url, data=data, method='PUT')
    req.add_header('Content-Type', 'application/octet-stream')
    with urllib.request.urlopen(req, timeout=600, context=ctx) as r:
        return r.status

def submit(file_path, message):
    c = KaggleClient()
    api = c.competitions.competition_api_client
    # 1. 申请上传
    req = ApiStartSubmissionUploadRequest()
    req.competition_name = COMP
    req.file_name = os.path.basename(file_path)
    req.content_length = os.path.getsize(file_path)
    req.last_modified_epoch_seconds = int(os.path.getmtime(file_path))
    resp = api.start_submission_upload(req)
    print('start upload OK, token:', resp.token[:40], '...')
    # 2. 上传到签名 URL
    st = upload_file_to_signed_url(file_path, resp.create_url)
    print('upload status:', st)
    # 3. 创建 submission
    sreq = ApiCreateSubmissionRequest()
    sreq.competition_name = COMP
    sreq.blob_file_tokens = resp.token
    sreq.submission_description = message
    sresp = api.create_submission(sreq)
    print('submission created, ref:', sresp.ref, '| message:', sresp.message)
    return sresp.ref

if __name__ == '__main__':
    f = sys.argv[1] if len(sys.argv) > 1 else 'output/submission.csv'
    m = sys.argv[2] if len(sys.argv) > 2 else 'auto_submit'
    submit(f, m)
