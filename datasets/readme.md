- txts: data in txt files.
  - One set sample per line, set elements are separated by `,`, and the beginning of the line can also contain the set name (separated by `:`).
    - `Ba,I,H,O`
    - `菠萝咕咾肉:猪肉,菠萝,青椒,红椒`
  - Set elements can be composed of element names and feature vectors (represented by `@[...]`), or they can only be composed of feature vectors.
    - `563-63-3:Ag@[1],C@[2],H@[3],O@[2]`
    - `@[1 2],@[2 3],@[3 4]`
- pkls: data in pickle files.

