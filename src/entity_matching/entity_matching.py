import entity_matching.utils.llm_utils as llm_utils
from pathlib import Path
from tqdm.asyncio import tqdm as async_tqdm
from tqdm import tqdm
import json
import re
import asyncio
import logging
from podbug.debug import Result_T, try_result

logger = logging.getLogger(__name__)

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

class BaseStemer:
    def __init__(self, model=None):
        self.word_cnt = 1
        self.word_index_dict = {}
        self.index_word_dict = {}

        self.word_definition_dict = {}
        self.api_model = model or 'deepseek-chat'

        self.parse_pattern = re.compile(r'```answer\s*(.*?)\s*```', re.DOTALL)

    def add(self, word, definition):
        if self.is_contained(word):
            return Result_T.error('word already exists')

        self.word_index_dict[word] = self.word_cnt
        self.word_definition_dict[word] = definition
        self.word_cnt += 1
        return Result_T.ok(self.word_index_dict[word])
    
    def add_dict(self, data_dict: dict):
        return [self.add(word, definition) for word, definition in data_dict.items()]
    
    def _matched(self, va, vb):
        return self.find(va).unwrap() != va or self.find(vb).unwrap() != vb
    
    def _parse_answer(self, completion: str):
        answer = self.parse_pattern.search(completion)
        if not answer:
            return None

        result = answer.group(1).strip('\n')
        logger.info(result)

        return result

    def _build(self):
        n = self.word_cnt
        self.fa = list(range(n))  # equivalent to std::iota
        self.index_word_dict = {val: key for key, val in self.word_index_dict.items()}

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

        self.fa[b] = a
        logger.info(f'{self.index_word_dict[a]} -> {self.index_word_dict[b]}')

        return Result_T()
    
    def find(self, x):
        if self.is_contained(x).is_err():
            return Result_T.error('no such word')
        
        return Result_T.ok(self.index_word_dict[self._find(self.word_index_dict[x])])

    def is_contained(self, word):
        return Result_T() if word in self.word_index_dict else Result_T.error('no such word')
    
    def _get_index(self, word):
        return Result_T.ok(self.word_index_dict[word]) if word in self.word_index_dict else Result_T.error('no such word')

    def _is_connected(self, a, b):
        a_index = self._get_index(a)
        b_index = self._get_index(b)
        if a_index.is_err() or b_index.is_err():
            return Result_T.error('no such word')

        return Result_T.ok(self._find(a_index.unwrap()) == self._find(b_index.unwrap()))
    
    def stem(self, word):
        if not self.is_contained(word):
            return Result_T.error(f'{word} is not included in database, use add() and build() again')
        
        return self.index_word_dict[self._find(self.word_index_dict[word])]

    def to_dict(self):
        result = {}

        for idx, e in enumerate(self.fa[1:]):
            if idx == e - 1:
                result[self.index_word_dict[e]] = []

        for idx, e in enumerate(self.fa[1:]):
            if idx != e - 1:
                result[self.index_word_dict[e]].append(self.index_word_dict[idx + 1])

        return result
    
    def save(self, to_file: Path | str):
        to_file = Path(to_file)

        with to_file.open('w', encoding='utf-8') as output_stream:
            json.dump([
                    {
                        'entity': entity,
                        'index': index,
                        'parent': self._find(self.fa[index]),
                        'definition': self.word_definition_dict[entity]
                    }
                    for entity, index in self.word_index_dict.items()
                ],
                output_stream,
                ensure_ascii=False,
            )

    def restore(self, from_file: Path | str):
        from_file = Path(from_file)

        with from_file.open('r', encoding='utf-8') as input_stream:
            data_json = json.load(input_stream)

        length = len(data_json)
        self.word_cnt = length + 1
        self.fa = list(range(length + 1))

        for data in data_json:
            entity = data['entity']
            index = data['index']
            parent = data['parent']
            definition = data['definition']

            self.word_index_dict[entity] = index
            self.index_word_dict[index] = entity
            self.fa[index] = parent
            self.word_definition_dict[entity] = definition


class Stemer(BaseStemer):
    def __init__(self, model=None):
        super().__init__(model)
    
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
            logger.error(f'cannot parse {completion}')
            return False
        
        return answer.upper() == "YES"
    
    def _link_to_most_similar(self):
        length = len(self.word_definition_dict)
        total_len = int(((length - 1) * length) / 2)

        with tqdm(total=total_len) as pbar:
            for i in range(0, length - 1):
                for j in range(i + 1, length):
                    va = self.index_word_dict[i + 1]
                    vb = self.index_word_dict[j + 1]

                    if (not self._matched(va, vb)) and self._entity_matching(va, vb):
                        self._merge(va, vb)

                    pbar.update()

    def build(self):
        self._build()
        self._link_to_most_similar()


class AStemer(BaseStemer):
    def __init__(self, model=None):
        super().__init__(model)
    
    async def _async_entity_matching(self, first_entity, second_entity):
        first_definition = self.word_definition_dict[first_entity]
        second_definition = self.word_definition_dict[second_entity]

        filled_prompt = PROMPT.format_map({
            'entity1': first_entity,
            'definition1': first_definition,
            'entity2': second_entity,
            'definition2': second_definition
        })

        completion = await llm_utils.openai_chat_completion(self.api_model, filled_prompt)
        answer = str(self._parse_answer(str(completion)))

        logger.info(f'{first_entity} and {second_entity} -> {answer}')

        if not answer:
            logger.info(f'cannot parse {completion}')
            return False, first_entity, second_entity
        
        return answer.upper() == "YES", first_entity, second_entity
    
    async def _async_link_to_most_similar(self):
        length = len(self.word_definition_dict)

        # Create all tasks for concurrent execution
        for i in range(0, length - 1):
            tasks = []

            for j in range(i + 1, length):
                va = self.index_word_dict[i + 1]
                vb = self.index_word_dict[j + 1]

                if not self._matched(va, vb):
                    tasks.append(self._async_entity_matching(va, vb))

            # Execute all tasks concurrently with progress tracking
            for coro in async_tqdm.as_completed(tasks, desc="Processing entity pairs"):
                should_match, entity_a, entity_b = await coro
                if should_match:
                    self._merge(entity_a, entity_b)

    async def abuild(self):
        n = self.word_cnt
        self.fa = list(range(n))  # equivalent to std::iota
        self.index_word_dict = {val: key for key, val in self.word_index_dict.items()}

        await self._async_link_to_most_similar()


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        filename="log/edc.log",
        filemode="a",
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )

    # async def async_main():
    #     ds = AStemer(Path('word_definition.jsonl'))

    #     data_dict = {
    #         '澳門': 'Refers to Macao, the Special Administrative Region of China and former Portuguese colony. A beautiful place',
    #         '濠鏡澳': 'An ancient Chinese name for Macao, literally meaning "Oyster Mirror Bay," referring to the area\'s geographic features before it became known as Macao.',
    #         '香港': 'refers to Hong Kong, the Special Administrative Region of China and former British colony.',
    #         '鏡海': 'refers to an ancient poetic name for the waters around Macao, literally meaning "Mirror Sea.'
    #     }

    #     ds.add_dict(data_dict)

    #     await ds.abuild()

    #     logger.info('standard representation of: ')
    #     for word in data_dict.keys():
    #         logger.info(f'{word} -> {ds.stem(word)}')

    #     logger.info(ds.to_dict())

    #     ds.save(Path('ds_state.json'))
    
    # def main():
    #     ds = Stemer(Path('word_definition.jsonl'))

    #     data_dict = {
    #         '澳門': 'Refers to Macao, the Special Administrative Region of China and former Portuguese colony. A beautiful place',
    #         '濠鏡澳': 'An ancient Chinese name for Macao, literally meaning "Oyster Mirror Bay," referring to the area\'s geographic features before it became known as Macao.',
    #         '香港': 'refers to Hong Kong, the Special Administrative Region of China and former British colony.',
    #         '鏡海': 'refers to an ancient poetic name for the waters around Macao, literally meaning "Mirror Sea.'
    #     }

    #     ds.add_dict(data_dict)

    #     ds.build()

    #     logger.info('standard representation of: ')
    #     logger.info(ds.to_dict())

    # asyncio.run(async_main())
    # main()

    ds = AStemer()
    ds.restore(Path('ds_state.json'))

    print(ds.stem('香港'))
