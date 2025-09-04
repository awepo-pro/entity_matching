import entity_matching.utils.llm_utils as llm_utils
from pathlib import Path
from tqdm.asyncio import tqdm as async_tqdm
from tqdm import tqdm
import json
import re
import asyncio
import logging
from podbug.debug import Result_T, try_result
# from pydantic import BaseModel, Field
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer

# class EntityMatchingInput(BaseModel):
#     entity: str
#     definition: str
#     metadata: dict = Field(default={}, description='the more metadata, the faster speed')

logger = logging.getLogger(__name__)

PROMPT = """ \
There are 1 main entity with defintion and several entity candidates' definition. You task is to identify are any of entity candidates refer to the same thing as main entity in reality? There are some examples, 

Example 1
<< Input >>
main entity: 澳門, is the modern Chinese name for Macau, referring to the Special Administrative Region of China.

candidate entity's definitions: 
1. is an old historical name for Macau, used during the Ming and Qing dynasties.
2. refers to Hong Kong, the Special Administrative Region of China and former British colony.
3. refers to HK
4. refers to an ancient poetic name for the waters around Macao, literally meaning "Mirror Sea.

<< Output >>
Looking at this example, I need to identify which candidate entities refer to the same real-world place as the main entity "澳門" (Macau).

**Analysis:**

Main entity: 澳門 - Modern Chinese name for Macau SAR, China

Candidate entities:
1. **Refers to the same entity** - This is an old historical name for Macau from the Ming and Qing dynasties. Despite being a different historical name, it refers to the same geographical location and political entity as modern 澳門.

2. **Does not refer to the same entity** - This refers to Hong Kong, which is a completely different Special Administrative Region of China, separate from Macau.

3. **Does not refer to the same entity** - This refers to HK (Hong Kong), same as candidate 2.

4. **Refers to the same entity** - This is an ancient poetic name for the waters around Macao. While it's a poetic/literary reference to the waters rather than the land territory itself, it refers to the same geographical area as 澳門.

```answer
YES
1
4
```

You might explain why you choose the answer as long as the explanation helps you to make better decision. Now, consider the following input,
<< Input >>
main_entity: {main_entity}, {main_definition}

candidate entity's definitions: 
{candidates}

"""

class BaseStemer:
    def __init__(self, model=None):
        self.word_cnt = 1
        self.word_index_dict = {}
        self.index_word_dict = {}

        self.word_definition_dict = {}
        self.api_model = model or 'deepseek-chat'

        self.FET_data_dict = {}
        self.word_FET_dict = {}

        self.parse_pattern = re.compile(r'```answer\s*(.*?)\s*```', re.DOTALL)
        self.api_call = 0

        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def add(self, word, FET, definition):
        if self.is_contained(word):
            return Result_T.error('word already exists')

        self.word_index_dict[word] = self.word_cnt
        self.word_definition_dict[word] = definition
        self.word_FET_dict[word] = FET
        if FET not in self.FET_data_dict:
            self.FET_data_dict[FET] = []

        self.FET_data_dict[FET].append((word, definition))
        self.word_cnt += 1

        return Result_T.ok(self.word_index_dict[word])
    
    def add_dict(self, data_dict: dict):
        return [self.add(word, v.get('FET', ''), v.get('definition', '')) for word, v in data_dict.items()]
    
    def _matched(self, va, vb):
        return self.find(va).unwrap() != va or self.find(vb).unwrap() != vb
    
    def _parse_answer(self, completion: str):
        answer = self.parse_pattern.search(completion)
        if not answer:
            return None, None

        result = answer.group(1).strip().split('\n')
        found, candidates = str(result[0]), list(result[1:])

        return found, candidates

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
        pass
        # first_definition = self.word_definition_dict[first_entity]
        # second_definition = self.word_definition_dict[second_entity]

        # filled_prompt = PROMPT.format_map({
        #     'entity1': first_entity,
        #     'definition1': first_definition,
        #     'entity2': second_entity,
        #     'definition2': second_definition
        # })

        # completion = str(llm_utils.fast_openai_chat_completion('deepseek-chat', filled_prompt))
        # answer = str(self._parse_answer(completion))

        # if not answer:
        #     logger.error(f'cannot parse {completion}')
        #     return False
        
        # return answer.upper() == "YES"
        # 
    def matching(self, similarity):
        threshold = 0.7
        length = len(similarity)
        result = []

        for i in range(0, length - 1, ):
            for j in range(i + 1, length):
                score = similarity[i][j]

                if score >= threshold:
                    result.append((i, j))

        return result
    
    def _link_to_most_similar(self, word_definition_list):
        input_def = [word_definition_tuple[1] for word_definition_tuple in word_definition_list]

        embeddings = self.model.encode(input_def)

        similarities = self.model.similarity(embeddings, embeddings)
        matched = self.matching(similarities)

        return matched
        
        # length = len(word_definition_list)

        # total_len = int(((length - 1) * length) / 2)

        # with tqdm(total=total_len) as pbar:
        #     for i in range(0, length - 1):
        #         for j in range(i + 1, length):
        #             va = word_definition_list[i][0]
        #             vb = word_definition_list[j][0]

        #             if (not self._matched(va, vb)) and self._entity_matching(va, vb):
        #                 self._merge(va, vb)

        #             pbar.update()

    def build(self):
        self._build()

        for word_definition_list in self.FET_data_dict.values():
            result = self._link_to_most_similar(word_definition_list)

            for result_tuple in result:
                print(f'{word_definition_list[result_tuple[0]][0]} <-> {word_definition_list[result_tuple[1]][0]}')

class AStemer(BaseStemer):
    def __init__(self, model=None):
        super().__init__(model)
    
    async def _async_entity_matching(self, main_entity, candidates):
        global cnt
        main_definition = self.word_definition_dict[main_entity]
        candidate_definition_list = [self.word_definition_dict[candidate] for candidate in candidates]

        candidates_str = '\n'.join(f'{idx}. {definition}' for idx, definition in enumerate(candidate_definition_list))

        filled_prompt = PROMPT.format_map({
            'main_entity': main_entity,
            'main_definition': main_definition,
            'candidates': candidates_str
        })

        self.api_call += 1
        completion = await llm_utils.openai_chat_completion(self.api_model, filled_prompt)
        found, answer = self._parse_answer(str(completion))

        if not found:
            logger.info(f'cannot parse {completion}')
            return False, main_definition, answer
        
        return found.upper() == "YES", main_entity, [candidates[int(i)] for i in answer] if answer else []
    
    async def _async_link_to_most_similar(self, word_definition_list):
        length = len(word_definition_list)

        # Create all tasks for concurrent execution
        for i in range(0, length - 1):
            tasks = []
            candidate_list = []
            va = word_definition_list[i][0]

            for j in range(i + 1, length):
                vb = word_definition_list[j][0]

                if not self._matched(va, vb):
                    candidate_list.append(vb)

            for i in range(0, len(candidate_list), 5):
                start = i
                end = i + 5
                candidate = candidate_list[start:end]
                tasks.append(self._async_entity_matching(va, candidate))

            # Execute all tasks concurrently with progress tracking
            for coro in async_tqdm.as_completed(tasks, desc="Processing entity pairs"):
                should_match, entity_a, entity_b = await coro
                if should_match:
                    for eb in entity_b:
                        self._merge(entity_a, eb)

    async def abuild(self):
        n = self.word_cnt
        self.fa = list(range(n))  # equivalent to std::iota
        self.index_word_dict = {val: key for key, val in self.word_index_dict.items()}

        def run_async_in_thread(self, word_definition_list):
            """Wrapper to run async function in a thread"""
            return asyncio.run(self._async_link_to_most_similar(word_definition_list))

        # Replace your code with:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(run_async_in_thread, self, word_definition_list)
                for word_definition_list in self.FET_data_dict.values()
            ]
            
            results = []
            for future in tqdm(as_completed(futures), total=len(futures), desc='processing blocking'):
                result = future.result()
                results.append(result)


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        filename="log/edc.log",
        filemode="a",
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )

    data_dict = {
        '澳門': {'FET': 'location', 'definition': 'Refers to Macao, the Special Administrative Region of China and former Portuguese colony. A beautiful place'},
        '濠鏡澳': {'FET': 'location', 'definition': 'An ancient Chinese name for Macao, literally meaning "Oyster Mirror Bay," referring to the area\'s geographic features before it became known as Macao.'},
        '香港': {'FET': 'location', 'definition': 'refers to Hong Kong, the Special Administrative Region of China and former British colony.'},
        'hk': {'FET': 'location', 'definition': 'refers to HK'},
        '鏡海': {'FET': 'location', 'definition': 'refers to an ancient poetic name for the waters around Macao, literally meaning "Mirror Sea.'},
        'pencil': {'FET': 'tool', 'definition': 'refer to a pen'},
        'pen': {'FET': 'tool', 'definition': 'refer to a pen'}
    }

    # data_dict: list[EntityMatchingInput] = [
    #     EntityMatchingInput(entity='澳門', definition='Refers to Macao, the Special Administrative Region of China and former Portuguese colony. A beautiful place', metadata={'FET': 'location'})
    # ]

    # async def async_main():
    #     ds = AStemer()

    #     ds.add_dict(data_dict)

    #     await ds.abuild()

    #     logger.info('standard representation of: ')
    #     for word in data_dict.keys():
    #         logger.info(f'{word} -> {ds.stem(word)}')

    #     logger.info(ds.to_dict())

    #     ds.save(Path('ds_state.json'))
    
    def main():
        ds = Stemer()

        ds.add_dict(data_dict)

        ds.build()

        logger.info('standard representation of: ')
        logger.info(ds.to_dict())

        ds.save(Path('ds_state.json'))

    main()

    # asyncio.run(async_main())
    # ds = AStemer()
    # ds.restore(Path('ds_state.json'))

    # print(ds.stem('香港'))
