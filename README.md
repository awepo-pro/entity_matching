# what is this

This is a package for solving entity matching problem. That is, given some entity (or whatever it is), we want to know if there are some entity refer to the same thing in reality. For example, given 

```python
data_dict = {
    '澳門': 'Refers to Macao, the Special Administrative Region of China and former Portuguese colony. A beautiful place',
    '濠鏡澳': 'An ancient Chinese name for Macao, literally meaning "Oyster Mirror Bay," referring to the area\'s geographic features before it became known as Macao.',
    '香港': 'refers to Hong Kong, the Special Administrative Region of China and former British colony.',
    '鏡海': 'refers to an ancient poetic name for the waters around Macao, literally meaning "Mirror Sea.'
}
```

It is obviously that `澳門, 濠鏡澳, 鏡海` refer to the same object - Macau. The core function `stem()` of this library is to return the `stem` of the entity within `data_dict`. In this case, 

```python
stem('澳門')        # 澳門
stem('濠鏡澳')      # 澳門
stem('香港')        # 香港
stem('鏡海')        # 澳門
```

# Example 

```python
import entity_matching as podstem
from pathlib import Path

ds = podstem.Stemer()     # init object  

data_dict = {
    '澳門': 'Refers to Macao, the Special Administrative Region of China and former Portuguese colony. A beautiful place',
    '濠鏡澳': 'An ancient Chinese name for Macao, literally meaning "Oyster Mirror Bay," referring to the area\'s geographic features before it became known as Macao.',
    '香港': 'refers to Hong Kong, the Special Administrative Region of China and former British colony.',
    '鏡海': 'refers to an ancient poetic name for the waters around Macao, literally meaning "Mirror Sea.'
}

ds.add_dict(data_dict)      # add data into object

ds.build()      # matching entity, it might take some times

for entity in data_dict.keys():     
    print(f'{entity} -> {ds.stem(entity)}')                     # print result

print('standard representation of: ' + str(ds.to_list()))       # print the result in list way
```

## asyncio
it also support asyncio, however, it might use more api call (since it cannot be optimized)

```python
import entity_matching as podstem
from pathlib import Path
import asyncio

async def main():
    ds = podstem.Stemer()     # init object  

    data_dict = {
        '澳門': 'Refers to Macao, the Special Administrative Region of China and former Portuguese colony. A beautiful place',
        '濠鏡澳': 'An ancient Chinese name for Macao, literally meaning "Oyster Mirror Bay," referring to the area\'s geographic features before it became known as Macao.',
        '香港': 'refers to Hong Kong, the Special Administrative Region of China and former British colony.',
        '鏡海': 'refers to an ancient poetic name for the waters around Macao, literally meaning "Mirror Sea.'
    }

    ds.add_dict(data_dict)      # add data into object

    await ds.abuild()      # matching entity, it might take some times

    for entity in data_dict.keys():     
        print(f'{entity} -> {ds.stem(entity)}}')                     # print result

    print('standard representation of: ' + str(ds.to_list()))       # print the result in list way

asyncio.run(main())
```

the only different is using `async/await` keyword and `abuild()` function instead of `build()`
