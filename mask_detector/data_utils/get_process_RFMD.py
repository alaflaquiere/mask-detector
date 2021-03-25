from pathlib import Path
import pandas as pd
from google_drive_downloader import GoogleDriveDownloader as gdd

RMFD_URL = "1UlOk6EtiaXTHylRUx2mySgvJX9ycoeBp"


def get_RFMD(root_p: Path = Path("data")):
    dataset_p = root_p / "self-built-masked-face-recognition-dataset"
    if dataset_p.exists():
        print("The dataset {} already exists".format(dataset_p))
    else:
        print("Dowloading and unzipping the dataset in {}".format(dataset_p))
        root_p.mkdir(parents=True)
        rar_p = root_p / "RMFD.rar"
        gdd.download_file_from_google_drive(file_id=RMFD_URL,
                                            dest_path=rar_p,
                                            unzip=True)
        rar_p.unlink()
    return dataset_p


def process_RFMD_as_df(dataset_p: Path = Path("data", "self-built-masked-face-recognition-dataset")):
    if not dataset_p.exists():
        print("{} doesn't exists".format(dataset_p))
        return None
    masked_p = dataset_p / "AFDB_masked_face_dataset"
    nonmasked_p = dataset_p / "AFDB_face_dataset"
    data_masked = [{"image_path": str(ip), "mask": 1}
                   for pers in masked_p.iterdir()
                   for ip in pers.iterdir()]
    data_nonmasked = [{"image_path": str(ip), "mask": 0}
                      for pers in nonmasked_p.iterdir()
                      for ip in pers.iterdir()]
    dataDf = pd.DataFrame(data_masked + data_nonmasked)

    pkl_p = dataset_p.parent / "RFMD_df.pkl"
    print("Saving dateframe in {}".format(pkl_p))
    dataDf.to_pickle(pkl_p)
    return dataDf


if __name__ == "__main__":
    dataset_p = get_RFMD(Path("data"))
    process_RFMD_as_df(dataset_p)
