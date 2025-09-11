import csv
import random
import re
import argparse

def fix_name(name: str) -> str:
    name = re.sub(r"\#\w+", "", name)
    try:
        second_block = re.search(r"/([^/]+)\.sdf$", name).group(1)
        fixed = re.sub(r"/([^/]+)_pocket", f"/{second_block}_pocket", name, count=1)
    except Exception:
        return name
    return fixed

def combine_text_dataset(src_files, dst_csv, use_columns, save_all):
    rows = []

    with open(src_files) as f_in:
        reader = csv.DictReader(f_in)
        for row in reader:
            available_cols = [col for col in use_columns if col in row and row[col].strip()]
            if row["text_func"] != "No prominent functional groups identified." and available_cols:
                if save_all:
                    for col in available_cols:
                        rows.append({"name": fix_name(row["name"]), "text": row[col]})
                else:
                    col = random.choice(available_cols)
                    rows.append({"name": fix_name(row["name"]), "text": row[col]})
            else:
                if save_all:
                    for col in available_cols:
                        rows.append({"name": fix_name(row["name"]), "text": row[col]})
                else:
                    available_cols = [acol for acol in available_cols if acol != "text_func"]
                    col = random.choice(available_cols)
                    rows.append({"name": fix_name(row["name"]), "text": row[col]})
    random.shuffle(rows)
    with open(dst_csv, "w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=["name", "text"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {dst_csv}")

def main():
    parser = argparse.ArgumentParser(description="Combine text dataset from multiple CSVs.")
    parser.add_argument('--src', required=True, help="Source CSV file(s)")
    parser.add_argument('--dst', required=True, help="Destination CSV file")
    parser.add_argument('--all', required=False, help="Use columns")
    args = parser.parse_args()

    # Possible source columns for text (extend if more were added)
    USE_COLUMNS = ["text_func", "text_llm_aug", "text_similar", "text_pharm_llm", "text_physchem_llm", "text_combined"]

    combine_text_dataset(args.src, args.dst, USE_COLUMNS, args.all)

if __name__ == "__main__":
    main()