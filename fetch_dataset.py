import os
from tokenizer import extract_text


def prep_dataset():
    """
    pre-process the dataset to remove html tags
    :return:
    """
    for root, dirs, files in os.walk('dataset'):
        prep_dataset = root.replace("dataset", "prep_dataset")
        if not os.path.exists(prep_dataset):
            os.makedirs(prep_dataset)

        for file in files:
            if "gitignore" not in file:
                path = os.path.join(root, file)
                text = extract_text(path)
                new_path = ".".join(path.split(".")[:-1])
                new_path = new_path.replace("dataset", "prep_dataset")+".txt"
                with open(new_path, 'w') as f:
                    f.write(text)


class Dataset:
    DESCR = ""
    data = []
    description = ""
    filenames = []
    target = []
    target_names = []


def fetch_data(subset="all"):
    """"
    subset: list of university names: ["cornell", "misc", "texas", "washington", "wisconsin"]
    Mimics fetch_20newsgroups api to load universities website data
    """
    if subset == "all":
        subset = ["cornell", "misc", "texas", "washington", "wisconsin"]

    dataset = Dataset()

    #load target names from os
    target_names = [item for item in os.listdir("dataset") if os.path.isdir(os.path.join("dataset", item))]
    data = []
    target = []

    #check if prepared dataset exists:
    root, dirs, files = os.walk("prep_dataset").__next__()
    
    if len(dirs) < 7:
        print(files)
        print(dirs)
        print("pre-processing dataset")
        prep_dataset()

    # walks through all files
    for root, dirs, files in os.walk('prep_dataset'):
        for file in files:
            # finds pre-processed files
            if "txt" in file:
                path = os.path.join(root, file)
                # checks target and target number
                target_name = path.split("/")[1]
                target_number = target_names.index(target_name)

                # checks if file is in selected subsets
                for item in subset:
                    if item in path.split(":")[0]:
                        with open(path) as f:
                            data.append(f.read())
                            target.append(target_number)

    # populates instance
    dataset.target_names = target_names
    dataset.target = target
    dataset.data = data
    return dataset
