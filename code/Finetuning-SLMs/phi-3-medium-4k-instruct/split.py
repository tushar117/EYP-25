from argparse import ArgumentParser
import os

def split_into_files(args):
    splits = args.n_splits
    res = [[] for _ in range(splits)]
    header = None
    with open(args.input_file, "r", encoding='utf-8') as dfile:
        for idx, line in enumerate(dfile.readlines()):
            if args.include_header and header is None:
                header = line.strip()
                for i in range(splits):
                    res[i].append(header)
                continue
            res[idx % splits].append(line.strip())

    # print len of each split
    for i in range(splits):
        print(f"Split {i+1} has {len(res[i])} records")

    # create a folder with name of the file without the extension
    args.input_file = os.path.abspath(args.input_file)
    base_dir = os.path.dirname(args.input_file)
    file_name = os.path.basename(args.input_file)
    folder_name = os.path.join(base_dir, file_name.split('.')[0])
    os.makedirs(folder_name, exist_ok=True)

    # remove any existing files/folders in the folder
    for f in os.listdir(folder_name):
        os.remove(os.path.join(folder_name, f))
    
    #write the splits to within the folder with _split{i}.{extension}
    extension = file_name.split('.')[-1]
    for i in range(splits):
        with open(os.path.join(folder_name, f"{file_name.split('.')[0]}_part_{i+1}.{extension}"), "w", encoding='utf-8') as dfile:
            for record in res[i]:
                dfile.write(f"{record}\n")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--include_header", action="store_true")
    parser.add_argument("--n_splits", type=int, default=4)
    args = parser.parse_args()
    split_into_files(args)