import yaml
from pathlib import Path

prompt_dict = {}
prompt_dir = Path('prompts')

# prompts 파일 내의 모든 .yml 파일 순회
for file_path in prompt_dir.glob("*.yml"):
    with file_path.open(encoding = "utf-8") as file:
        fname = file_path.stem   # 확장자 제외한 부분 반환
        prompt_dict[fname] = yaml.safe_load(file)   # 파일 내용 로드하여 prompt_dict에 저장
    