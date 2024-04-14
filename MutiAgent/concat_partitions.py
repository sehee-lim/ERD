import pandas as pd

folder1 = "240318-135841"
folder2 = "240318-135849"
folder3 = "240318-135855"
folder4 = "240318-135900"
folder5 = "240318-135906"
folder6 = "240318-135912"
folder7 = "240318-135917"
folder8 = "240318-135922"
folder9 = "240318-135928"
folder10 = "240318-135933"
folders = [folder1,
           folder2,
           folder3,
           folder4,
           folder5,
           folder6, folder7, folder8, folder9, folder10]


result = pd.read_csv(f"output/{folder1}/SubsetIDX-1-exp_result.csv")
for i in range(2, 11):
    directory = f"output/{folders[i-1]}/SubsetIDX-{i}-exp_result.csv"
    df = pd.read_csv(directory)
    result = pd.concat([result, df])
    print(i)

result.to_csv(f"output/{folder1}/SubsetIDX-1_full-exp_result.csv")

# pd.set_option('display.max_rows', None)
# result["Distortion Type"].value_counts()

# types = ["All-or-nothing thinking", "Overgeneralization", "Mental filter", "Should statements", "Labeling", "Personalization", "Magnification", "Emotional Reasoning", "Mind Reading", "Fortune-telling", "No distortion"]
# types = [x.lower() for x in types]

# result["Distortion Type"] = result["Distortion Type"].str.lower()
# result.loc[result["Distortion Assessment"] == False, "Distortion Type"] = "no distortion"
# result_valid = result[result["Distortion Type"].isin(types)]
# # len(result_valid)
# result_valid.to_csv(f"output/{folder2}/SubsetIDX-240125_full_valid-exp_result.csv")

