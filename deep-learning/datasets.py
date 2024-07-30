import matplotlib.pyplot as plt
import os
import glob
from monai.transforms import Compose, LoadImaged
from monai.data import Dataset
from monai.config import PathLike
import json
from pathlib import Path


def generate_misa_json_datalist(
    base_dir: PathLike | None = None, output_file: PathLike = "misa_datalist.json"
) -> None:
    """
    Generate a json file with the image/label filenames of the MISA final project dataset.
    Args:
        base_dir: the base directory of the dataset.
        output_file: the file name of the output json file.

    """
    # normalize the base directory
    base_dir = os.path.normpath(base_dir)

    # get all the filenames recursively
    train_filenames = glob.glob(
        os.path.join(base_dir, "Training_Set", "**", "*.nii.gz"), recursive=True
    )
    val_filenames = glob.glob(
        os.path.join(base_dir, "Validation_Set", "**", "*.nii.gz"), recursive=True
    )
    test_filenames = glob.glob(
        os.path.join(base_dir, "Test_Set", "**", "*.nii.gz"), recursive=True
    )

    # keep only the two last folder names and the filename
    train_filenames = [
        os.path.join(
            os.path.basename(os.path.dirname(os.path.dirname(filename))),
            os.path.basename(os.path.dirname(filename)),
            os.path.basename(filename),
        )
        for filename in train_filenames
    ]
    val_filenames = [
        os.path.join(
            os.path.basename(os.path.dirname(os.path.dirname(filename))),
            os.path.basename(os.path.dirname(filename)),
            os.path.basename(filename),
        )
        for filename in val_filenames
    ]
    test_filenames = [
        os.path.join(
            os.path.basename(os.path.dirname(os.path.dirname(filename))),
            os.path.basename(os.path.dirname(filename)),
            os.path.basename(filename),
        )
        for filename in test_filenames
    ]

    # for the testing set, create a list of dictionaries with only the image filenames
    test_dict = [{"image": filename} for filename in test_filenames]

    # separate segmentation files
    train_seg_filenames = [
        filename for filename in train_filenames if "seg" in filename
    ]
    train_image_filenames = [
        filename for filename in train_filenames if "seg" not in filename
    ]
    val_seg_filenames = [filename for filename in val_filenames if "seg" in filename]
    val_image_filenames = [
        filename for filename in val_filenames if "seg" not in filename
    ]

    # add the both image and segmentation filenames to a list of dictionaries
    train_dict = [
        {"image": image_filename, "label": seg_filename}
        for image_filename, seg_filename in zip(
            train_image_filenames, train_seg_filenames
        )
    ]
    val_dict = [
        {"image": image_filename, "label": seg_filename}
        for image_filename, seg_filename in zip(val_image_filenames, val_seg_filenames)
    ]

    # create a dictionary with test, train and validation sets
    final_dict = {
        "testing": test_dict,
        "training": train_dict,
        "validation": val_dict,
    }

    # save the dictionary as a json file
    with open(os.path.join(base_dir, output_file), "w") as fp:
        json.dump(final_dict, fp)


def load_misa_datalist(
    data_list_file_path: PathLike,
    data_list_key: str = "training",
    base_dir: PathLike | None = None,
) -> list[dict]:
    """
    Load image/label paths of MISA final project dataset.

    Args:
        data_list_file_path: the path to the json file of datalist.
        base_dir: the base directory of the dataset, if None, use the datalist directory.
        data_list_key: the key to get a list of dictionary to be used, default is "training".
            Other keys can be: "validation", "testing".

    Raises:
        ValueError: When ``data_list_file_path`` does not point to a file.
        ValueError: When ``data_list_key`` is not specified in the data list file.

    Returns a list of data items, each of which is a dict keyed by element names, for example:

    .. code-block::

        [
            {'image': '/workspace/data/IBSR_01/IBSR_01.nii.gz',  'label': '/workspace/data/IBSR_01/IBSR_01_seg.nii.gz'},
            {'image': '/workspace/data/IBSR_03/IBSR_03.nii.gz',  'label': '/workspace/data/IBSR_03/IBSR_03_seg.nii.gz'}
        ]

    """
    data_list_file_path = Path(data_list_file_path)
    if not data_list_file_path.is_file():
        raise ValueError(f"Data list file {data_list_file_path} does not exist.")
    with open(data_list_file_path) as json_file:
        json_data = json.load(json_file)
    if data_list_key not in json_data:
        raise ValueError(
            f'Data list {data_list_key} not specified in "{data_list_file_path}".'
        )
    expected_data = json_data[data_list_key]

    if base_dir is None:
        base_dir = data_list_file_path.parent

    for item in expected_data:
        for key in item:
            item[key] = os.path.join(base_dir, item[key])

    return expected_data


if __name__ == "__main__":
    # Test run to visualize the dataset

    data_dir = (
        "d:\\Files\\GDrive\\Documents\\MaIA\\Courses\\UdG\\MISA\\Project\\dataset"
    )

    json_file_path = os.path.join(data_dir, "misa_dataset.json")

    if not os.path.exists(json_file_path):
        generate_misa_json_datalist(data_dir, "misa_dataset.json")

    datalist = load_misa_datalist(
        data_list_file_path=json_file_path,
        data_list_key="training",
    )

    train_transforms = Compose(
        [LoadImaged(keys=["image", "label"], ensure_channel_first=True)]
    )

    dataset = Dataset(data=datalist, transform=train_transforms)

    # simulate a single iteration of a for loop
    batch = next(iter(dataset))

    print(batch["image"].shape)

    # plot slice 150 of the first image in the batch and its segmentation
    plt.figure()
    plt.subplot(121)
    plt.imshow(batch["image"][0, :, :, 150])
    plt.subplot(122)
    plt.imshow(batch["label"][0, :, :, 150])
    plt.show()
