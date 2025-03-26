import os 

def get_filenames(directory_path: str) -> list[str]:
  files = []
  for item in os.listdir(directory_path):
    item_path = os.path.join(directory_path, item)
    if os.path.isfile(item_path) and item[0] != ".":
      files.append(item_path)
    elif os.path.isdir(item_path):
        files += get_filenames(directory_path + "/" + item) 
  return files

def get_original_source(filename: str, prefix: str) -> str:
    if prefix in filename.split('/')[-1][0:len(prefix)]:
        return get_original_source(filename.split('/')[-1][len(prefix):], prefix)
    else:
        return filename


def reconcile_duplicates(prefix:str="") -> None:
    evaluated = set()
    for filename in get_filenames('./datasets/evals/'):
        if  prefix in filename.split('/')[-1][0:len(prefix)]:
            og = get_original_source(filename, prefix)
            if og in evaluated:
                os.remove(filename)
            else:
                evaluated.add(og)


reconcile_duplicates("optimized_")
