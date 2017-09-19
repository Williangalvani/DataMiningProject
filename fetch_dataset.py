import os, shutil, random
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
			#if "gitignore" not in file:
				path = os.path.join(root, file)
				text = extract_text(path)
				if path.endswith(".html"):
						new_path = ".".join(path.split(".")[:-1])
				else:
						new_path = path
				new_path = new_path.replace("dataset", "prep_dataset")+".txt"
				with open(new_path, 'w') as f:
					f.write(text)


def temp_dataset():
	"""
	remove the universities folder from the tree
	"""
	if not os.path.exists("temp_dataset"):
		os.makedirs("temp_dataset")

	dirs = [d for d in os.listdir("prep_dataset") if os.path.isdir(os.path.join("prep_dataset", d))]
	for directory in dirs:
		if not os.path.exists(os.path.join("temp_dataset", directory)):
			os.makedirs(os.path.join("temp_dataset", directory))
		for university in os.listdir(os.path.join("prep_dataset", directory)):
			for file in os.listdir(os.path.join("prep_dataset", directory, university)):
				shutil.copy(os.path.join("prep_dataset", directory, university, file), os.path.join("temp_dataset", directory))


def split_dataset(percentage=0.2):
	"""
	separates trainning and testing files
	"""

	temp_dataset()

	train_path = os.path.join("split_dataset","train")
	test_path = os.path.join("split_dataset","test")

	if not os.path.exists("split_dataset"):
		os.makedirs("split_dataset")

	if not os.path.exists(os.path.join("split_dataset","train")):
		os.makedirs(train_path)

	if not os.path.exists(os.path.join("split_dataset","test")):
		os.makedirs(test_path)

	if os.path.exists("temp_dataset"):
		classes = [d for d in os.listdir("temp_dataset") if os.path.isdir(os.path.join("temp_dataset", d))]
		for classe in classes:
			temp_path = os.path.join("temp_dataset", classe)
			test_class_path = os.path.join(test_path, classe)
			train_class_path = os.path.join(train_path, classe)

			if not os.path.exists(test_class_path):
				os.makedirs(test_class_path)

			if not os.path.exists(train_class_path):
				os.makedirs(train_class_path)
			
			fileList = os.listdir(temp_path)

			for _ in range(round(percentage*len(fileList))):
				randomFile = random.choice(fileList)
				shutil.copy(os.path.join(temp_path, randomFile), test_class_path)
				fileList.remove(randomFile)

			for file in fileList:
				shutil.copy(os.path.join(temp_path, file), train_class_path)

	if os.path.exists("temp_dataset"):
		shutil.rmtree("temp_dataset")



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
		print("converting html files to txt")
		prep_dataset()

	# walks through all files
	for root, dirs, files in os.walk('prep_dataset'):
		for file in files:
			# finds pre-processed files
			if "txt" in file and "gitignore" not in file:
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


def fetch_train_data(subset="all"):
	""""
	subset: list of classess: ["course", "department", "faculty", "other", "project", "staff", "student"]
	Mimics fetch_20newsgroups api to load universities website data
	"""
	if subset == "all":
		subset = ["course", "department", "faculty", "other", "project", "staff", "student"]

	dataset = Dataset()

	#load target names from os
	target_names = [item for item in os.listdir("dataset") if os.path.isdir(os.path.join("dataset", item))]
	data = []
	target = []

	# walks through all files
	for root, dirs, files in os.walk(os.path.join("split_dataset","train")):
		for file in files:
			# finds pre-processed files
			if "txt" in file and "gitignore" not in file:
				path = os.path.join(root, file)
				# checks target and target number
				target_name = path.split("/")[2]
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


def fetch_test_data(subset="all"):
	""""
	subset: list of classess: ["course", "department", "faculty", "other", "project", "staff", "student"]
	Mimics fetch_20newsgroups api to load universities website data
	"""
	if subset == "all":
		subset = ["course", "department", "faculty", "other", "project", "staff", "student"]

	dataset = Dataset()

	#load target names from os
	target_names = [item for item in os.listdir("dataset") if os.path.isdir(os.path.join("dataset", item))]
	data = []
	target = []

	# walks through all files
	for root, dirs, files in os.walk(os.path.join("split_dataset","test")):
		for file in files:
			# finds pre-processed files
			if "txt" in file and "gitignore" not in file:
				path = os.path.join(root, file)
				# checks target and target number
				target_name = path.split("/")[2]
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
