import argparse
import os
from glob import glob
from typing import Iterable
import pandas as pd
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString

REQUIRED_COLUMNS = [
    "_rlnCoordinateX",
    "_rlnCoordinateY",
    "_rlnCoordinateZ",
]
ANGLE_COLUMNS = [
    "_rlnAngleRot",
    "_rlnAngleTilt",
    "_rlnAnglePsi",
]


def parse_star_file(file_path: str, require_angles: bool = True) -> pd.DataFrame:
    """
    Parse a RELION-style .star file into a DataFrame.

    Expects a single data block with column definitions and rows.
    """
    with open(file_path, "r") as star_file:
        lines = [line.strip() for line in star_file if line.strip() and not line.strip().startswith("#")]

    columns = []
    data_rows = []
    in_data = False

    for line in lines:
        if line.startswith("loop_"):
            columns = []
            in_data = False
            continue
        if line.startswith("_rln"):
            columns.append(line.split()[0])
            continue
        if columns:
            in_data = True
        if in_data:
            row = line.split()
            if len(row) >= len(columns):
                data_rows.append(row[:len(columns)])

    if not columns:
        raise ValueError(f"No _rln columns found in {file_path}")

    df = pd.DataFrame(data_rows, columns=columns)
    required = list(REQUIRED_COLUMNS)
    if require_angles:
        required += ANGLE_COLUMNS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {file_path}: {missing}")

    return df


def dataframe_to_xml(
    df: pd.DataFrame,
    class_label: int,
    tomo_name: str,
    include_angles: bool = True,
) -> str:
    root = Element("objlist")
    for _, row in df.iterrows():
        obj_elem = SubElement(root, "object")
        obj_elem.set("tomo_name", str(tomo_name))
        obj_elem.set("class_label", str(class_label))
        obj_elem.set("x", str(row["_rlnCoordinateX"]))
        obj_elem.set("y", str(row["_rlnCoordinateY"]))
        obj_elem.set("z", str(row["_rlnCoordinateZ"]))
        if include_angles:
            obj_elem.set("phi", str(row["_rlnAngleRot"]))
            obj_elem.set("the", str(row["_rlnAngleTilt"]))
            obj_elem.set("psi", str(row["_rlnAnglePsi"]))
    return parseString(tostring(root)).toprettyxml(indent="  ")


def _combine_xml_blocks(xml_blocks: Iterable[str]) -> str:
    xml_blocks = [block for block in xml_blocks if block.strip()]
    if not xml_blocks:
        return parseString(tostring(Element("objlist"))).toprettyxml(indent="  ")

    combined = xml_blocks[0].split("\n")
    for block in xml_blocks[1:]:
        combined = (
            combined[:-2]
            + block.split("\n")[2:-1]
            + [combined[-1]]
        )
    return "\n".join(combined)


def expand_inputs(inputs: list[str]) -> list[str]:
    expanded = []
    for item in inputs:
        if os.path.isdir(item):
            expanded.extend(sorted(glob(os.path.join(item, "*.star"))))
        else:
            matches = glob(item)
            expanded.extend(sorted(matches) if matches else [item])
    return expanded


def parse_class_map(items: list[str]) -> dict:
    mapping = {}
    for item in items:
        if ":" not in item:
            raise ValueError(f"Invalid --class-map entry (expected name:label): {item}")
        name, label = item.split(":", 1)
        mapping[name] = int(label)
    return mapping


def main():
    parser = argparse.ArgumentParser(description="Convert RELION .star annotations to EMPIAR-style XML.")
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Input .star files (or directories/globs containing .star files).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output XML file path (e.g., tomo0001_objl.xml). If omitted, uses --tomo-name.",
    )
    parser.add_argument(
        "--class-labels",
        nargs="+",
        type=int,
        default=None,
        help="Class labels aligned with --input order (default: 1..N).",
    )
    parser.add_argument(
        "--class-map",
        nargs="+",
        default=None,
        help="Mapping from input basename to class label, e.g. classA:1 classB:2.",
    )
    parser.add_argument(
        "--tomo-name",
        default=None,
        help="Tomogram name used in XML and default output name (e.g., tomo0001).",
    )
    parser.add_argument(
        "--no-angles",
        action="store_true",
        help="Skip angle columns and omit phi/the/psi in XML.",
    )
    args = parser.parse_args()

    inputs = expand_inputs(args.input)
    if not inputs:
        raise ValueError("No input .star files found.")

    class_map = parse_class_map(args.class_map) if args.class_map else {}

    if args.class_labels is not None and len(args.class_labels) != len(inputs):
        raise ValueError(
            f"--class-labels expects {len(inputs)} values, got {len(args.class_labels)}"
        )

    class_labels = []
    for idx, path in enumerate(inputs):
        name = os.path.splitext(os.path.basename(path))[0]
        if name in class_map:
            class_labels.append(class_map[name])
        elif args.class_labels is not None:
            class_labels.append(args.class_labels[idx])
        elif class_map:
            raise ValueError(f"Missing class map entry for input: {name}")
        else:
            class_labels.append(idx + 1)

    if args.output is None:
        if not args.tomo_name:
            raise ValueError("--output is required when --tomo-name is not provided.")
        output_path = f"{args.tomo_name}_objl.xml"
    else:
        output_path = args.output

    tomo_name = args.tomo_name or "tomo0"
    include_angles = not args.no_angles

    xml_blocks = []
    for star_path, class_label in zip(inputs, class_labels):
        df = parse_star_file(star_path, require_angles=include_angles)
        xml_blocks.append(dataframe_to_xml(df, class_label, tomo_name, include_angles=include_angles))

    combined_xml = _combine_xml_blocks(xml_blocks)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as xml_file:
        xml_file.write(combined_xml)

    print(f"Wrote XML with {len(inputs)} class block(s) to {output_path}")


if __name__ == "__main__":
    main()
