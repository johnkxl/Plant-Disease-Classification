# ######################################################################
# CSCI 444 Machine Learning
# Advanced Project Phase 2
# 
# John Kendall
# Gabriel Gaspar
# Zephyr Zamora
# 
# Script for cleaning and formatting the New Plant Diseases Dataset
# for use with df-embed
# 
# ######################################################################

# usage: python create-images-df.py <dataset-path> <leaf-maps-path>


from io import BytesIO
from PIL import Image
from pandas import DataFrame
import os
import sys
from pathlib import Path
import json
from csv import DictReader


def main() -> int:
    DATASET_PATH = Path(sys.argv[1])
    LEAF_MAPS_PATH = Path(sys.argv[2])

    # Only images that are able to be associated with a leaf_ID are included
    df = load_dataframe(DATASET_PATH, LEAF_MAPS_PATH)

    # Final format of df
    # plant_ID : INTEGER unique for each subject
    # full_name : STRING unique
    # file_name : STRING unique
    # image : BYTES
    # plant : STRING plant type 
    # plant_disease : STRING 
    # disease : STRING
    # disease_binary : INTEGER BINARY (1 or 0)
    
    save_all_files(df)

    # Format of DataFrames saved in csv files:
    # plant_ID : INTEGER
    # target : STRING

    return 0


def load_dataframe(dataset_path: Path, leaf_maps_path: Path) -> DataFrame:
    """Returns DataFrame containing images loaded from `dataset_path`."""
    labels_dir  = os.listdir(dataset_path)

    plant_types = []  # Plant Types
    plant_disease = []  # Plant_Disease pairs
    particular_diseases = []  # Just disease names
    img_names = []  # Unique names to associate images to a single subject
    file_names = []

    # Identify plant and diseaase classes from filenames
    for label in labels_dir:
        plant, disease = label.split("___")

        label_path = dataset_path / label
        images_with_label = os.listdir(label_path)

        for file in images_with_label:
            plant_disease.append(label)
            plant_types.append(plant)
            particular_diseases.append(disease)
            file_names.append(file)

            img_name = file
            if img_name.__contains__("___"):
                img_name = file.split("___")[1]
            img_names.append(img_name)

    table = {
        # "image": converted,
        "plant": plant_types,
        "plant_disease": plant_disease,
        "disease": particular_diseases,
        # "disease_binary": disease_binary,
        "img_name": img_names,
        "full_name": file_names
    }
    df = DataFrame(table)
    df = clean_dataset(df, plant_types, leaf_maps_path)

    # Add Image bytes to corresponding records
    # This is done last to reduce the total amount of Image conversions
    img: Image  # PIL Image
    converted = []
    indices = []
    for row in df.itertuples():
        img_path = dataset_path / Path(f"{row.plant_disease}/{row.full_name}")
        img = Image.open(img_path)
        buf = BytesIO()
        img.save(buf, format="JPEG")
        byts = buf.getvalue()
        converted.append(byts)
        indices.append(row.full_name)
    
    image_df = DataFrame({"full_name":indices, "image": converted})
    df = df.join(image_df.set_index("full_name"), on="full_name")
    return df


def save_all_files(df: DataFrame) -> None:
    # Create a directory to hold all the configurations of the dataset.
    all_parquets_dir: Path = Path("all_parquet_configs")
    os.makedirs(all_parquets_dir, exist_ok=True)

    # Create parquet files for configurations involving all plants.
    subset_name = "all_plants"
    all_plants_dir = all_parquets_dir / subset_name
    os.makedirs(all_plants_dir, exist_ok=True)
    save_image_subset_config_files(df, subset_name, all_plants_dir, ["plant_disease", "disease_binary"])
    
    # Create parquet files for configurations for each plant
    plant_types = list(df["plant"].unique())
    by_plant_dir = all_parquets_dir / "by_plant"
    os.makedirs(by_plant_dir, exist_ok=True)
    plant_df: DataFrame
    for plant in plant_types:
        plant_df = df[df["plant"] == plant]
        save_image_subset_config_files(plant_df, plant, by_plant_dir, ["disease", "disease_binary"])
    return


def generate_single_target_df(df: DataFrame, target: str) -> DataFrame:
    """Return a pandas DataFrame with only "image" and "target" columns."""
    img_target = df[["image", target]]
    img_target = img_target.rename(columns={target: "target"})
    return img_target


def create_class_integer_mapping(df: DataFrame) -> dict[str, int]:
    """Return a dict that maps all classes in the target column to integers starting at 0."""
    classes = list(df["target"].unique())
    mapping = {value: i for i, value in enumerate(classes)}
    return mapping


def map_classess_to_integers(df: DataFrame, mapping: dict[str, int]) -> DataFrame:
    """Return a DataFrame with values in column "target" mapped according to mapping."""
    int_targets_df = df.copy()
    int_targets_df["target"] = int_targets_df["target"].map(mapping)
    return int_targets_df


def save_parquet_and_mapping(config_name: str, df_config_dir: Path, one_target_df: DataFrame, mapping: dict[str, int]) -> None:
    """
    Saves the one_target_df Dataframe as a parquet file alongside mapping dictionary as a JSON file is the df_config_dir directory.

    one_target_df
     DataFrame containing two columns: "images" and "target."

    mapping 
     Dictionary that maps classes to their integer values represented in the "target" column.
    """
    parquet_name = f"{config_name}.parquet"
    one_target_df.to_parquet(df_config_dir / parquet_name, index=False)
    print(f"\t\tCreated {parquet_name}.")
    
    json_name = f"{config_name}_mapping.json"
    json_mapping_file = df_config_dir / json_name
    
    # Create integer: class dictionary from class: integer dictionary
    int_class_dict = {int(v):k for k,v in mapping.items()}
    with open(json_mapping_file, "w") as f:
        json.dump(int_class_dict, f, default=str)
    
    print(f"\t\tCreated {json_name}")
    return


def save_image_subset_config_files(img_df: DataFrame, subset_name: str, subset_dir: Path, target_list: list[str]) -> None:
    """
    Save configurations of subsets of the image dataset as parquet files in the subset directory.

    img_df
     subset of the image dataset.
    
    subset_name
     name of the subset.

    subset_dir
     directory to house all confiurations of the subset.

    target_list
     list of targets to create image-target configurations.
    """
    print(f"Creating Dataframes for subset {subset_name}...")
    one_target_df: DataFrame
    mapping: dict[str, int]
    for target in target_list:
        config_name = f"{subset_name}_{target}"
        print(f"\tDataset configuration: {config_name}")
        df_config_dir = subset_dir / config_name

        one_target_df = generate_single_target_df(img_df, target)
        mapping = create_class_integer_mapping(one_target_df)
        one_target_df = map_classess_to_integers(one_target_df, mapping)
        os.makedirs(df_config_dir, exist_ok=True)
        save_parquet_and_mapping(config_name, df_config_dir, one_target_df, mapping)

        # Save Dataframe as csv containing leaf_ID and target variable for association after embedding
        csv_name = f"{config_name}.csv"
        temp_df = img_df[["leaf_ID", target]]
        temp_df.to_csv(df_config_dir / csv_name, index=False)
        print(f"\t\tCreated {csv_name}.")

    return


def clean_dataset(df: DataFrame, plant_types: list[str], leaf_maps_path: Path) -> DataFrame:
    """Return cleaned DataFrame."""
    # Drop plants that only have one disease class.
    temp_df: DataFrame
    plants_to_drop = []
    for plant in plant_types:
        temp_df = df[df["plant"] == plant]
        if temp_df["disease"].nunique() < 2:
            plants_to_drop.append(plant)  
    df = df[~df['plant'].isin(plants_to_drop)]
    
    # Create DataFrame that maps image names to unique leaves
    leaf_map_lst = os.listdir(leaf_maps_path)
    data_dict: dict[str,list] = {}
    reader: DictReader
    for file_name in leaf_map_lst:
        file_path = leaf_maps_path / file_name
        with open(file_path, mode="r") as file:
            reader = DictReader(file)
            # Initialise keys for first file
            for column in reader.fieldnames:
                if column not in data_dict.keys():
                    data_dict[column] = []
            # Append rows to dict.
            for row in reader:
                for column in data_dict.keys():
                    data_dict[column].append(row[column])

    unique_leaves_df = DataFrame(data_dict)

    # Left join unique_leaves_df to df to identify groups of images of same leaf
    df = unique_leaves_df.join(df.set_index("img_name"), on="File Name")
    df = df.dropna()
    
    # Rename columns for ease of use.
    df = df.rename(columns={"Leaf #": "leaf_num", "File Name": "file_name"})

    # Must find all unique plant-leaf_num combinations since leaf_num is only
    # unique for each plant-disease pair.
    unique_plant_leaf_num = set()
    for row in df.itertuples():
        unique_plant_leaf_num.add((row.plant, row.leaf_num, row.disease))
    
    # Create lead_ID column with a unique value for each plant-leaf_num-disease
    df["leaf_ID"] = None
    i = 0
    for plant, leaf_num, disease in unique_plant_leaf_num:
        df.loc[(df["plant"] == plant) 
                    & (df["leaf_num"] == leaf_num) 
                    & (df["disease"] == disease), "leaf_ID"] = i
        i += 1
    df = df.drop(columns=["leaf_num"])

    # Create disease_binary now that the DataFrame contains the least rows.
    df["disease_binary"] = [0] * df.shape[0]
    df.loc[df["disease"] == "healthy", "disease_binary"] = 1
    return df



if __name__ == "__main__":
    main()