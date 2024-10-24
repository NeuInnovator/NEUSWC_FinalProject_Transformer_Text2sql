#导入必要的功能模块
import json
import os
import datasets
import config

# 本地数据路径
DATA_DIR = config.DATA_WIKISQL
# sql查询中的聚合操作
_AGG_OPS = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
# sql查询中的条件操作符
_COND_OPS = ["=", ">", "<", "OP"]

# WikiSQL 继承了 datasets.GeneratorBasedBuilder，用于定义自定义数据集加载逻辑。
class WikiSQL(datasets.GeneratorBasedBuilder):
    """WikiSQL: A large crowd-sourced dataset for developing natural language interfaces for relational databases"""
    # 定义了数据集的版本
    VERSION = datasets.Version("0.1.0")

# 定义数据集的元信息，包括字段的结构和类型
    def _info(self):
        return datasets.DatasetInfo(
            description="A large crowd-sourced dataset for developing natural language interfaces for relational databases",
            features=datasets.Features(
                {
                    # 训练/验证/测试的阶段编号
                    "phase": datasets.Value("int32"),
                    # 自然语言问题
                    "question": datasets.Value("string"),
                    # 包含表格的相关信息（标题、页码、列名、行数据等）
                    "table": {
                        "header": datasets.features.Sequence(datasets.Value("string")),
                        "page_title": datasets.Value("string"),
                        "page_id": datasets.Value("string"),
                        "types": datasets.features.Sequence(datasets.Value("string")),
                        "id": datasets.Value("string"),
                        "section_title": datasets.Value("string"),
                        "caption": datasets.Value("string"),
                        "rows": datasets.features.Sequence(datasets.features.Sequence(datasets.Value("string"))),
                        "name": datasets.Value("string"),
                    },
                    # SQL查询的信息，包括选择列 (sel)、聚合操作 (agg)、条件 (conds) 等
                    "sql": {
                        "human_readable": datasets.Value("string"),
                        "sel": datasets.Value("int32"),
                        "agg": datasets.Value("int32"),
                        "conds": datasets.features.Sequence(
                            {
                                "column_index": datasets.Value("int32"),
                                "operator_index": datasets.Value("int32"),
                                "condition": datasets.Value("string"),
                            }
                        ),
                    },
                }
            ),
            homepage="https://github.com/salesforce/WikiSQL",
            citation="...citation details..."
        )

    # 定义数据集的拆分方式
    def _split_generators(self, dl_manager):
        """Returns SplitGenerators based on local files."""
        # 训练集，验证集，测试集数组，定义三个集的拆分方式，使用datasets.SplitGenerator将数据集文件路径传递给_generate_examples()函数进行解析
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "main_filepath": os.path.join(DATA_DIR, "test.jsonl"),
                    "tables_filepath": os.path.join(DATA_DIR, "test.tables.jsonl"),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "main_filepath": os.path.join(DATA_DIR, "dev.jsonl"),
                    "tables_filepath": os.path.join(DATA_DIR, "dev.tables.jsonl"),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "main_filepath": os.path.join(DATA_DIR, "train.jsonl"),
                    "tables_filepath": os.path.join(DATA_DIR, "train.tables.jsonl"),
                },
            ),
        ]

    # 将SQL查询转换为人类可读的字符串形式，使用聚合操作和条件操作符构建查询
    def _convert_to_human_readable(self, sel, agg, columns, conditions):
        """Make SQL query string."""
        rep = f"SELECT {_AGG_OPS[agg]} {columns[sel]} FROM table"
        if conditions:
            rep += " WHERE " + " AND ".join([f"{columns[i]} {_COND_OPS[o]} {v}" for i, o, v in conditions])
        return rep

    # 解析JSONL文件并生成样本
    def _generate_examples(self, main_filepath, tables_filepath):
        # 打开表格文件，将每个表格的id映射到表格数据
        with open(tables_filepath, encoding="utf-8") as f:
            tables = [json.loads(line) for line in f]
            id_to_tables = {x["id"]: x for x in tables}

        # 打开主要数据文件，为每条记录补充对应的表格信息
        with open(main_filepath, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                row = json.loads(line)
                row["table"] = id_to_tables[row["table_id"]]
                del row["table_id"]

                row["table"]["page_title"] = row["table"].get("page_title", "")
                row["table"]["section_title"] = row["table"].get("section_title", "")
                row["table"]["caption"] = row["table"].get("caption", "")
                row["table"]["name"] = row["table"].get("name", "")
                row["table"]["page_id"] = str(row["table"].get("page_id", ""))

                # 将表格中的所有数据都强制转换为字符串类型，并重新存储在row["table"]["rows"]中
                row["table"]["rows"] = [[str(e) for e in r] for r in row["table"]["rows"]]

                # 将SQL查询转换为人类可读的形式
                row["sql"]["human_readable"] = self._convert_to_human_readable(
                    row["sql"]["sel"],
                    row["sql"]["agg"],
                    row["table"]["header"],
                    row["sql"]["conds"]
                )

                # 重新格式化SQL查询中的条件数据
                for i in range(len(row["sql"]["conds"])):
                    row["sql"]["conds"][i] = {
                        "column_index": row["sql"]["conds"][i][0],
                        "operator_index": row["sql"]["conds"][i][1],
                        "condition": str(row["sql"]["conds"][i][2]),
                    }

                # 使用yield 生成每条数据
                yield idx, row
