import pandas as pd

full = pd.read_csv("./data/Annotated_data.csv")
full["User speech"] = full["Patient Question"]   # input으로 Patient Question 열을 사용

num_partitions = 10   # 전체 데이터를 10개로 split
partition_size = len(full) // num_partitions

for i in range(num_partitions):
    # 파티션 생성
    partition = full[i * partition_size:(i + 1) * partition_size]

    # 파티션을 csv 파일로 저장
    partition.to_csv(f'./data/subset-{i + 1}.csv', index = False)

print("All partition files have been saved.")