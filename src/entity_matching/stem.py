from pydantic import BaseModel, Field
from typing import Any
import edc.utils.llm_utils as llm_utils
from pathlib import Path
from tqdm.asyncio import tqdm as async_tqdm
from tqdm import tqdm
import json
import re
import asyncio

PROMPT = """ \
There are 2 entities and their defintions, are they refer to the same thing in reality? There are some examples, 
Example 1
<< Input >>
entity 1: 澳門, is the modern Chinese name for Macau, referring to the Special Administrative Region of China.
entity 2: 濠鏡澳, is an old historical name for Macau, used during the Ming and Qing dynasties.

<< Output >>
Both terms refer to the same geographic location—the peninsula and islands we know today as Macau.

The difference is purely historical and stylistic: 澳門 (Àomén) is the modern, official name, while 濠鏡澳 (Háojìng'ào) is an ancient, literary name that is rarely used in everyday modern contexts.

```answer
YES
```

Example 2
<< Input >>
entity 1: 澳門, is the modern Chinese name for Macau, referring to the Special Administrative Region of China.
entity 2: 香港, refers to Hong Kong, the Special Administrative Region of China and former British colony.

<< Output >>
These are two completely different geographic locations and political entities. While both are Special Administrative Regions of China with similar political status, they are:

Different territories with distinct boundaries
Different historical backgrounds (Portuguese vs British colonial heritage)
Different locations (Macau is west of the Pearl River Delta, Hong Kong is to the east)
Different cultures, languages, and administrative systems
Different economies and legal frameworks

```answer
NO
```

You might explain why you choose the answer as long as the explanation helps you to make better decision. Now, consider the following input,
<< Input >>
entity 1: {entity1}, {definition1}
entity 2: {entity2}, {definition2}

<< Output >>

"""

class Result_T(BaseModel):
    data: Any = Field(default=None)
    err: str | None = Field(default=None)

    def is_ok(self):
        return not self.err

    def is_err(self):
        return self.err is not None

    def __str__(self):
        """String representation"""
        if self.err is not None:
            return f"Result(err='{self.err}')"
        return str(self.data)
    
    def __repr__(self):
        """Debug representation"""
        if self.err is not None:
            return f"Result(err='{self.err}')"
        return f"Result(data={repr(self.data)})"
    
    def __bool__(self):
        """Boolean evaluation - True if no error"""
        return self.err is None
    
    def unwrap(self):
        """Get the data - only use when you're sure there's no error"""
        return self.data
    
    def unwrap_or(self, default):
        """Get the data or return default if there's an error"""
        if self.err is not None:
            return default
        return self.data
    
    def and_then(self, func) -> "Result_T":
        """Chain operations - only applies func if no error"""
        if self.err is not None:
            return self
        try:
            return func(self.data)
        except Exception as e:
            return Result_T.error(str(e))
        
    @classmethod
    def ok(cls, data) -> "Result_T":
        """Create a successful result"""
        return cls(data=data)
    
    @classmethod
    def error(cls, err: str) -> "Result_T":
        """Create an error result"""
        return cls(err=err)
    
    def __getitem__(self, key):
        return self.data[key]
    
    def __len__(self):
        return len(self.data) if self.data is not None else 0
    
def try_result(operation):
    try:
        if callable(operation):
            # call it if it's callable
            return Result_T.ok(operation())
        else:
            # if it's already a Result_T, return as-is
            return operation if isinstance(operation, Result_T) else Result_T.ok(operation)
    except Exception as e:
        return Result_T.error(str(e))

class Stemer:
    def __init__(self, threshold, word_definition_db_path: Path | None=None, verbose=False):
        self.word_cnt = 1
        self.word_index_dict = {}
        self.index_word_dict = {}

        self.word_definition_dict = {}
        self.threshold = threshold
        self.word_definition_db_path = word_definition_db_path or Path('word_definition.jsonl')
        self.verbose = verbose

        self.parse_pattern = re.compile(r'```answer\s*(.*?)\s*```', re.DOTALL)

        self._restore_word_definition()

    def __del__(self):
        self._store_word_definition()

    def _store_word_definition(self):
        with self.word_definition_db_path.open('w', encoding='utf-8') as output:
            for word, definition in self.word_definition_dict.items():
                output.write(json.dumps({
                    'word': word,
                    'definition': definition
                }, ensure_ascii=False) + '\n')

    def _restore_word_definition(self):
        if not (self.word_definition_db_path and self.word_definition_db_path.exists()):
            return 
        
        backup_word_definition_dict = {}
        
        with self.word_definition_db_path.open('r', encoding='utf-8') as input:
            for line in input:
                jl: Result_T = try_result(json.loads(line))

                if not jl:
                    print(f'Error: fail to pass {line}')
                    print(f'The exact error is {jl.err}')
                    continue
                
                backup_word_definition_dict[jl['word']] = jl['definition']

        self.add_dict(backup_word_definition_dict)

    def add(self, word, definition):
        if self.is_contained(word):
            return Result_T.error('word already exists')

        self.word_index_dict[word] = self.word_cnt
        self.word_definition_dict[word] = definition
        self.word_cnt += 1
        return Result_T.ok(self.word_index_dict[word])
    
    def add_dict(self, data_dict: dict):
        return [self.add(word, definition) for word, definition in data_dict.items()]
    
    async def _async_link_to_most_similar(self):
        async def _async_entity_matching(first_entity, second_entity):
            first_definition = self.word_definition_dict[first_entity]
            second_definition = self.word_definition_dict[second_entity]

            filled_prompt = PROMPT.format_map({
                'entity1': first_entity,
                'definition1': first_definition,
                'entity2': second_entity,
                'definition2': second_definition
            })

            completion = await llm_utils.openai_chat_completion('deepseek-chat', filled_prompt)
            answer = str(self._parse_answer(str(completion)))

            print(f'{first_entity} and {second_entity} -> {answer}')

            if not answer:
                print(f'cannot parse {completion}')
                return False, first_entity, second_entity
            
            return answer.upper() == "YES", first_entity, second_entity

        length = len(self.word_definition_dict)
        
        # Create all tasks for concurrent execution
        tasks = []
        for i in range(0, length - 1):
            for j in range(i + 1, length):
                va = self.index_word_dict[i + 1]
                vb = self.index_word_dict[j + 1]
                tasks.append(_async_entity_matching(va, vb))
        
        # Execute all tasks concurrently with progress tracking
        for coro in async_tqdm.as_completed(tasks, desc="Processing entity pairs"):
            should_match, entity_a, entity_b = await coro
            if should_match:
                self._merge(entity_a, entity_b)

    async def abuild(self):
        n = self.word_cnt
        self.fa = list(range(n))  # equivalent to std::iota
        self.size = [1] * n
        self.index_word_dict = {val: key for key, val in self.word_index_dict.items()}

        await self._async_link_to_most_similar()

    def _parse_answer(self, completion: str):
        answer = self.parse_pattern.search(completion)
        if not answer:
            return None

        return answer.group(1).strip()

    def _entity_matching(self, first_entity, second_entity):
        first_definition = self.word_definition_dict[first_entity]
        second_definition = self.word_definition_dict[second_entity]

        filled_prompt = PROMPT.format_map({
            'entity1': first_entity,
            'definition1': first_definition,
            'entity2': second_entity,
            'definition2': second_definition
        })

        completion = str(llm_utils.fast_openai_chat_completion('deepseek-chat', filled_prompt))
        answer = str(self._parse_answer(completion))

        if not answer:
            print(f'cannot parse {completion}')
            return False
        
        return answer.upper() == "YES"

    def _link_to_most_similar(self):
        length = len(self.word_definition_dict)

        for i in tqdm(range(0, length - 1)):
            for j in tqdm(range(i + 1, length)):
                va = self.index_word_dict[i + 1]
                vb = self.index_word_dict[j + 1]

                if self._entity_matching(va, vb):
                    self._merge(va, vb)
                    break

    def build(self):
        n = self.word_cnt
        self.fa = list(range(n))  # equivalent to std::iota
        self.size = [1] * n
        self.index_word_dict = {val: key for key, val in self.word_index_dict.items()}

        self._link_to_most_similar()


    # amortized to O(1) for path compression and weighted-union heuristic
    def _find(self, x):
        if self.fa[x] != x:
            self.fa[x] = self._find(self.fa[x])  # path compression
        return self.fa[x]
    
    def _merge(self, a, b):
        a_index = self._get_index(a)
        b_index = self._get_index(b)

        if a_index.is_err() or b_index.is_err():
            return Result_T.error('no such word')

        a = self._find(a_index.unwrap())
        b = self._find(b_index.unwrap())
        if a == b:
            return Result_T()
        

        # weighted-union heuristic: attach smaller tree under root of larger tree
        if self.size[a] <= self.size[b]:
            a, b = b, a  # swap
        
        self.size[a] += self.size[b]
        self.fa[b] = a
        print(f'{self.index_word_dict[a]} -> {self.index_word_dict[b]}') if self.verbose else None

        return Result_T()
    
    def find(self, x):
        if self.is_contained(x).is_err():
            return Result_T.error('no such word')
        
        return Result_T.ok(self._find(self.word_index_dict[x]))

    def is_contained(self, word):
        return Result_T() if word in self.word_index_dict else Result_T.error('no such word')
    
    def _get_index(self, word):
        return Result_T.ok(self.word_index_dict[word]) if word in self.word_index_dict else Result_T.error('no such word')

    def is_connected(self, a, b):
        a_index = self._get_index(a)
        b_index = self._get_index(b)
        if a_index.is_err() or b_index.is_err():
            return Result_T.error('no such word')

        return Result_T.ok(self._find(a_index.unwrap()) == self._find(b_index.unwrap()))
    
    def stem(self, word):
        if not self.is_contained(word):
            return Result_T.error(f'{word} is not included in database, use add() and build() again')
        
        return self.index_word_dict[self._find(self.word_index_dict[word])]

    def to_list(self):
        result = {}

        for idx, e in enumerate(self.fa[1:]):
            if idx == e - 1:
                result[self.index_word_dict[e]] = []

        for idx, e in enumerate(self.fa[1:]):
            if idx != e - 1:
                result[self.index_word_dict[e]].append(self.index_word_dict[idx + 1])

        return result


async def async_main():
    ds = Stemer(0.8, Path('word_definition.jsonl'), verbose=True)

    data_dict = {
        '澳門': 'Refers to Macao, the Special Administrative Region of China and former Portuguese colony. A beautiful place',
        '濠鏡澳': 'An ancient Chinese name for Macao, literally meaning "Oyster Mirror Bay," referring to the area\'s geographic features before it became known as Macao.',
        # '香港': 'refers to Hong Kong, the Special Administrative Region of China and former British colony.',
        # '鏡海': 'refers to an ancient poetic name for the waters around Macao, literally meaning "Mirror Sea.'
    }

    ds.add_dict(data_dict)

    await ds.abuild()

    print('standard representation of: ')
    for word in data_dict.keys():
        print(f'{word} -> {ds.stem(word)}')

    print(ds.to_list())
    
def main():
    ds = Stemer(0.8, Path('word_definition.jsonl'), verbose=True)

    data_dict = {
        '澳門': 'Refers to Macao, the Special Administrative Region of China and former Portuguese colony. A beautiful place',
        '濠鏡澳': 'An ancient Chinese name for Macao, literally meaning "Oyster Mirror Bay," referring to the area\'s geographic features before it became known as Macao.',
        # '香港': 'refers to Hong Kong, the Special Administrative Region of China and former British colony.',
        # '鏡海': 'refers to an ancient poetic name for the waters around Macao, literally meaning "Mirror Sea.'
    }

    ds.add_dict(data_dict)

    ds.build()

    print('standard representation of: ')
    print(ds.to_list())

if __name__ == "__main__":
    import logging

    logging.basicConfig(
        filename="log/edc.log",
        filemode="a",
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )

    asyncio.run(async_main())
