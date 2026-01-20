import pandas as pd
import os
import re
import subprocess
import shutil
import hashlib
import glob
import csv  # 用于严格 CSV 输出控制

# ================= 配置区域 =================
INPUT_CSV_DIR = "./original/File-level"
CK_JAR_PATH = "./ck-0.7.0-jar-with-dependencies.jar"
TEMP_SRC_ROOT = "./output/temp_src"
OUTPUT_METRICS_DIR = "./output/baseline_data_mbl_final_v14"  # V14
# ===========================================

# ===========================================
# 核心：完全使用你规定的 SHA1 计算方式
# ===========================================
def canonical_src(src: str) -> str:
    if not isinstance(src, str):
        return ""
    s = src.replace("\r\n", "\n").replace("\r", "\n")
    lines = s.split("\n")
    lines = [ln.rstrip() for ln in lines]
    return "\n".join(lines)

def sha1_of_src(src: str) -> str:
    canon = canonical_src(src)
    try:
        return hashlib.sha1(canon.encode("utf-8"), usedforsecurity=False).hexdigest()
    except TypeError:
        return hashlib.sha1(canon.encode("utf-8")).hexdigest()
# ===========================================

def normalize_path(path):
    return os.path.normpath(os.path.abspath(path)).lower()

def parse_filename(filename):
    base = os.path.basename(filename)
    core_name = base.replace("_ground-truth-files_dataset.csv", "").replace(".csv", "")
    if "-" in core_name:
        parts = core_name.rsplit("-", 1)
        return parts[0], parts[1]
    return core_name, "unknown"

def extract_class_name(source_code):
    match = re.search(r'public\s+class\s+(\w+)', source_code)
    if match:
        return match.group(1)
    match = re.search(r'class\s+(\w+)', source_code)
    if match:
        return match.group(1)
    return None

def fix_ck_output_location(target_dir):
    target_dir = os.path.abspath(target_dir)
    expected_file = os.path.join(target_dir, "class.csv")
    if os.path.exists(expected_file):
        return True

    malformed_path = target_dir + "class.csv"
    if os.path.exists(malformed_path):
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        try:
            shutil.move(malformed_path, expected_file)
            for junk in ["method.csv", "field.csv", "variable.csv"]:
                junk_p = target_dir + junk
                if os.path.exists(junk_p):
                    os.remove(junk_p)
            return True
        except:
            return False

    cwd_file = os.path.join(os.getcwd(), "class.csv")
    if os.path.exists(cwd_file):
        shutil.move(cwd_file, expected_file)
        return True
    return False

def get_representative_class_name(group_df, filename_no_ext):
    """定位策略：优先文件名同名，拒绝 $"""
    candidates = group_df[~group_df['class'].astype(str).str.contains(r'\$', case=False, na=False)]
    if candidates.empty:
        candidates = group_df

    candidates = candidates.copy()
    candidates['short_name'] = candidates['class'].astype(str).apply(lambda x: x.split('.')[-1])

    perfect_match = candidates[candidates['short_name'].str.lower() == str(filename_no_ext).lower()]
    if not perfect_match.empty:
        return perfect_match.iloc[0]['class']

    # 兜底：挑 loc 最大的那个
    if 'loc' in candidates.columns:
        return candidates.sort_values(by='loc', ascending=False).iloc[0]['class']
    return candidates.iloc[0]['class']

def aggregate_ck_metrics(group_df):
    """聚合策略：Sum/Max/Mean（仅对数值列生效；非数值列自动忽略）"""
    agg_rules = {}
    cols_sum = [
        'loc', 'wmc', 'totalMethodsQty', 'staticMethodsQty', 'publicMethodsQty',
        'privateMethodsQty', 'protectedMethodsQty', 'defaultMethodsQty', 'visibleMethodsQty',
        'abstractMethodsQty', 'finalMethodsQty', 'synchronizedMethodsQty',
        'totalFieldsQty', 'staticFieldsQty', 'publicFieldsQty', 'privateFieldsQty',
        'protectedFieldsQty', 'defaultFieldsQty', 'finalMethodsQty', 'finalFieldsQty',
        'synchronizedFieldsQty', 'nosi', 'returnQty', 'loopQty', 'comparisonsQty',
        'tryCatchQty', 'parenthesizedExpsQty', 'stringLiteralsQty', 'numbersQty',
        'assignmentsQty', 'mathOperationsQty', 'variablesQty', 'anonymousClassesQty',
        'innerClassesQty', 'lambdasQty', 'uniqueWordsQty', 'logStatementsQty'
    ]
    cols_max = ['cbo', 'cboModified', 'fanin', 'fanout', 'rfc', 'dit', 'noc', 'maxNestedBlocksQty']
    cols_mean = ['lcom', 'lcom*', 'tcc', 'lcc']

    for c in group_df.columns:
        if c in cols_sum:
            agg_rules[c] = 'sum'
        elif c in cols_max:
            agg_rules[c] = 'max'
        elif c in cols_mean:
            agg_rules[c] = 'mean'
        else:
            if pd.api.types.is_numeric_dtype(group_df[c]):
                agg_rules[c] = 'max'

    return group_df.groupby(lambda x: 0).agg(agg_rules).iloc[0]

def _strip_bom_columns(df: pd.DataFrame) -> pd.DataFrame:
    # 读取时统一去 BOM：\ufeffProject 这种
    df = df.copy()
    df.columns = [str(c).lstrip('\ufeff') for c in df.columns]
    return df

def _pick_column(df: pd.DataFrame, candidates_lower):
    """按候选列名（小写）从 df 中找真实列名"""
    lower_map = {str(c).lower(): c for c in df.columns}
    for k in candidates_lower:
        if k in lower_map:
            return lower_map[k]
    return None

def _parse_label(v):
    """统一 label 解析：输出 0/1（支持 0/1、true/false、yes/no 等）"""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return 0
    s = str(v).strip().lower()
    if s in ["1", "true", "yes", "y", "t"]:
        return 1
    if s in ["0", "false", "no", "n", "f"]:
        return 0
    # 尝试数值化兜底
    try:
        return 1 if float(s) != 0.0 else 0
    except:
        return 0

def _robust_read_csv(path):
    import csv as pycsv
    import sys

    # python 引擎字段过长时会炸：提高上限（Windows 上用 2**31-1 最稳）
    try:
        pycsv.field_size_limit(2**31 - 1)
    except Exception:
        pass

    if not os.path.exists(path):
        print(f"  [DEBUG] CSV 路径不存在: {path}")
        return None

    sz = os.path.getsize(path)
    if sz == 0:
        print(f"  [DEBUG] CSV 文件大小为 0: {path}")
        return None

    last_err = None

    # 1) 优先 C 引擎（旧脚本行为），更适合超长 SRC
    for enc in ["utf-8", "utf-8-sig", "latin1", "cp1252"]:
        try:
            df = pd.read_csv(path, encoding=enc, engine="c", low_memory=False)
            df = _strip_bom_columns(df)
            return df
        except Exception as e:
            last_err = (enc, "c", repr(e))

    # 2) 再尝试 python 引擎（必要时），并把错误打印出来
    for enc in ["utf-8", "utf-8-sig", "latin1", "cp1252"]:
        try:
            df = pd.read_csv(path, encoding=enc, engine="python")
            df = _strip_bom_columns(df)
            return df
        except Exception as e:
            last_err = (enc, "python", repr(e))

    print(f"  [DEBUG] CSV 无法读取: {path} (size={sz} bytes), last_err={last_err}")
    return None


def process_single_csv(csv_path, project_metrics_buffer):
    project_name, version = parse_filename(csv_path)
    print(f"\n 处理: [{project_name}] - Ver: [{version}]")

    version_root_dir = os.path.join(TEMP_SRC_ROOT, project_name, version)
    ck_out_dir = os.path.join(TEMP_SRC_ROOT, project_name, f"{version}_ck_out")

    if os.path.exists(version_root_dir):
        shutil.rmtree(version_root_dir)
    os.makedirs(version_root_dir, exist_ok=True)

    if os.path.exists(ck_out_dir):
        shutil.rmtree(ck_out_dir)
    os.makedirs(ck_out_dir, exist_ok=True)

    # --- 读取 ---
    df = _robust_read_csv(csv_path)
    if df is None or df.empty:
        print(f"  [DEBUG] CSV 文件为空或无法读取: {csv_path}")
        return

    print(f"  [DEBUG] 读取到 {len(df)} 行数据")

    # --- 列名定位（大小写/别名兼容）---
    src_col = _pick_column(df, ["src"])  # 只认 SRC/src（你要求最终不输出，但输入必须靠它还原）
    label_col = _pick_column(df, ["label", "target"])
    bug_col = _pick_column(df, ["bug", "bugs", "is_buggy", "defect"])

    if src_col is None:
        print("  [WARN] 未找到 SRC/src 列，跳过该文件")
        print(f"  [DEBUG] 可用列名: {list(df.columns)}")
        return

    print(f"  [DEBUG] 找到 src 列: {src_col}")

    # --- 还原源码 -> 文件（仅用于 CK 计算，不会输出到最终 CSV） ---
    file_map = {}
    for idx, row in df.iterrows():
        src = row.get(src_col, "")
        if pd.isna(src) or not str(src).strip():
            continue
        src = str(src)

        # label 统一：优先 label/target，其次 bug/bugs
        raw_label = row.get(label_col, None) if label_col else (row.get(bug_col, 0) if bug_col else 0)
        label = _parse_label(raw_label)

        sha1 = sha1_of_src(src)
        class_name = extract_class_name(src) or f"File_{idx}"

        fname = f"{class_name}.java"
        fpath = os.path.join(version_root_dir, fname)
        if os.path.exists(fpath):
            fpath = os.path.join(version_root_dir, f"{class_name}_{idx}.java")

        try:
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(src)
            norm_path = normalize_path(fpath)
            file_map[norm_path] = {
                "Project": project_name,
                "Version": version,
                "SHA1": sha1,
                "label": label,
                "filename_no_ext": class_name
            }
        except Exception as e:
            print(f"  [DEBUG] 写入文件失败: {fpath}, 错误: {str(e)}")
            continue

    print(f"  [DEBUG] 创建了 {len(file_map)} 个源码文件")
    
    if not file_map:
        print(f"  [DEBUG] 没有创建任何源码文件，跳过此版本")
        return

    # --- CK ---
    abs_jar = os.path.abspath(CK_JAR_PATH)
    abs_src = os.path.abspath(version_root_dir)
    abs_out = os.path.abspath(ck_out_dir)
    cmd = ["java", "-jar", abs_jar, abs_src, "false", "0", "false", abs_out]
    print(f"  [DEBUG] 执行 CK 分析命令")
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if result.returncode != 0:
            print(f"  [DEBUG] CK 工具执行失败，返回码: {result.returncode}")
            print(f"  [DEBUG] 错误输出: {result.stderr.decode('utf-8', errors='ignore')[:500]}...")
    except Exception as e:
        print(f"  [DEBUG] 执行 CK 工具时发生异常: {str(e)}")
        return

    if not fix_ck_output_location(abs_out):
        print("  [WARN] 未生成 class.csv")
        return

    # --- 读取 CK 输出 ---
    ck_csv_path = os.path.join(abs_out, "class.csv")
    try:
        ck_df = pd.read_csv(ck_csv_path, engine="python")
        ck_df = _strip_bom_columns(ck_df)
        print(f"  [DEBUG] 读取到 CK 输出 {len(ck_df)} 行数据")
    except Exception as e:
        print(f"  [DEBUG] 读取 CK 输出失败: {str(e)}")
        return

    # 关键：只让数值列保持数值（非数值列后续会被丢弃，避免写出 NaN/类型混乱）
    for c in ck_df.columns:
        if c in ["file", "class"]:
            continue
        ck_df[c] = pd.to_numeric(ck_df[c], errors="coerce")

    ck_df["norm_path"] = ck_df["file"].apply(normalize_path)
    valid_rows = ck_df[ck_df["norm_path"].isin(file_map)]
    print(f"  [DEBUG] 找到 {len(valid_rows)} 个有效行")
    
    if valid_rows.empty:
        print(f"  [DEBUG] 没有匹配的 CK 结果")
        return

    count = 0
    grouped = valid_rows.groupby("norm_path")

    for norm_path, group_df in grouped:
        meta = file_map[norm_path]
        rep_class_name = get_representative_class_name(group_df, meta["filename_no_ext"])
        agg_metrics = aggregate_ck_metrics(group_df)

        combined = meta.copy()
        combined["class"] = rep_class_name
        combined["file"] = norm_path
        combined.update(agg_metrics.to_dict())

        for k in ["norm_path", "filename_no_ext", "short_name"]:
            combined.pop(k, None)

        project_metrics_buffer.setdefault(project_name, []).append(combined)
        count += 1

    print(f"    成功提取并聚合: {count} 个文件")

def _finalize_and_write_project_csv(proj, data, out_dir):
    df = pd.DataFrame(data)
    if df.empty:
        return 0, None

    # 去 BOM（极端情况）
    df = _strip_bom_columns(df)

    # 硬检查：确保 label 一定存在（否则跳过）
    if "label" not in df.columns:
        print(f"[WARN] {proj}: missing label column -> skip writing merged.csv")
        return 0, None

    # label 强制 0/1 int（可解析一致）
    df["label"] = df["label"].apply(_parse_label).astype(int)

    # 使用一致 SHA1 去重
    if "SHA1" in df.columns:
        df = df.drop_duplicates(subset=["SHA1"], keep="first")

    # 确保 Project/Version 字段存在且非空
    if "Project" not in df.columns:
        df["Project"] = proj
    df["Project"] = df["Project"].fillna("unknown").astype(str).apply(lambda x: x.strip() or "unknown")

    if "Version" not in df.columns:
        df["Version"] = "unknown"
    df["Version"] = df["Version"].fillna("unknown").astype(str).apply(lambda x: x.strip() or "unknown")

    # 确保 file/class 不为空
    for col in ["file", "class"]:
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype(str).apply(lambda x: x.strip() or "unknown")

    # 只保留：meta + 数值特征 + label
    meta_cols = ["Project", "Version", "file", "class", "SHA1"]
    keep_meta = [c for c in meta_cols if c in df.columns]

    # 识别“真正的数值特征列”
    numeric_cols = []
    for c in df.columns:
        if c in keep_meta or c == "label":
            continue
        # 强制转数值，非数值变 NaN，后续仍保留该列但它就是数值列了（避免字符串列混进来）
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)

    # 列顺序固定：meta + 数值特征(排序) + label
    numeric_cols = sorted(numeric_cols)
    final_cols = keep_meta + numeric_cols + (["label"] if "label" in df.columns else [])
    df_final = df[final_cols]

    # 缺失值：统一写空（或你也可改成 na_rep="NaN"）
    output_path = os.path.join(out_dir, f"{proj}_merged.csv")

    # 严格 CSV：逗号分隔、字段必要时双引号包裹、内部 " -> ""、每行列数一致
    df_final.to_csv(
        output_path,
        index=False,
        encoding="utf-8",          # 禁止 BOM（不要 utf-8-sig）
        sep=",",
        na_rep="",
        quoting=csv.QUOTE_MINIMAL,
        quotechar='"',
        doublequote=True,
        lineterminator="\n"
    )
    return len(df_final), output_path

def main():
    if not os.path.exists(CK_JAR_PATH):
        print(f"[ERROR] CK jar not found: {CK_JAR_PATH}")
        return

    csv_files = glob.glob(os.path.join(INPUT_CSV_DIR, "*.csv"))
    project_metrics_buffer = {}

    print(f" 开始处理 {len(csv_files)} 个版本...")
    for f in csv_files:
        process_single_csv(f, project_metrics_buffer)

    if not project_metrics_buffer:
        print("[WARN] No metrics extracted.")
        return

    if os.path.exists(OUTPUT_METRICS_DIR):
        shutil.rmtree(OUTPUT_METRICS_DIR)
    os.makedirs(OUTPUT_METRICS_DIR, exist_ok=True)

    print("\n 正在保存最终结果（无 SRC，严格 CSV）...")
    for proj, data in project_metrics_buffer.items():
        n, p = _finalize_and_write_project_csv(proj, data, OUTPUT_METRICS_DIR)
        if p:
            print(f"   -> {proj}: 保存 {n} 条  ({p})")

if __name__ == "__main__":
    main()
