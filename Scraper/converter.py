import jsonlines
from tree_sitter import Language, Parser
from datasets import load_dataset

Language.build_library(
    # Store the library in the `build` directory
    'build/my-languages.so',

    # Include one or more languages
    [
        'tree-sitter-java'
    ]
)
JAVA_LANGUAGE = Language('build/my-languages.so', 'java')
parser = Parser()
parser.set_language(JAVA_LANGUAGE)


lines = []
with jsonlines.open('D:\Programming\codebert\GraphCodeBERT\codesearch\dataset\java\german_doc_and_func.jsonl') as reader:
    for obj in reader:
        func_name = None
        test = obj["function"]
        tree = parser.parse(bytes(test, "utf8"))
        for node in tree.root_node.named_children:
            if node.type != "local_variable_declaration":
                break
            for subnode in node.named_children:
                if subnode.type == "variable_declarator":
                    func_name = subnode.text.decode("utf-8")

        if func_name is None:
            continue
        lines.append({'repo': obj['nwo'],
                      'path': obj['path'],
                      'func_name': obj['identifier'] + func_name,
                      'original_string': obj['function'],
                      'language': obj['language'],
                      'code': obj['function'],
                      'code_tokens': obj['function_tokens'],
                      'docstring': obj['docstring'],
                      'docstring_tokens': obj['docstring_tokens'],
                      'sha': obj['sha'],
                      'url': obj['url'],
                      'partition': "dev+test"})


with jsonlines.open('repo_data/converted_german_doc_and_func.jsonl', mode="w") as writer:
    writer.write_all(lines)
