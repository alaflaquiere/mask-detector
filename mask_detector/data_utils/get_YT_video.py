from argparse import ArgumentParser
from pathlib import Path
import pytube


def get_YT_video(url: str, save_p: Path):
    print("Downloading video from: {}".format(url))
    youtube = pytube.YouTube(url)
    video = youtube.streams.get_highest_resolution()
    save_p.mkdir(parents=True, exist_ok=True)
    video.download(save_p)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-u",
                        default="https://www.youtube.com/watch?v=2TTg53aAP8Q",
                        help="YouTube video url")
    parser.add_argument("-o",
                        default=Path("data", "video"),
                        help="output directory")
    args = parser.parse_args()

    get_YT_video(args.u, args.o)
