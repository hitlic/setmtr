import numpy as np
import pickle
import random as rand
from functools import partial
from itertools import chain
from torchvision.datasets import MNIST
from torchvision import transforms
import open3d as o3d

img_size = 0

def load_from_txt(file_path, min_set_size=2, element_dict=None):
    """
    Load set dataset from a text file with diferent format.
        when ele_name=True and feat_dim=0:  Ba,I,H,O                  -> The set element is id
        when ele_name=True and feat_dim>0:  Ag@[1 2],C@[2 3], H@[3 4] -> The set element is [id, feat_vec]
        when ele_name=False and feat_dim>0: @[1 2],@[2 3], @[3 4]     -> The set element is feat_vec
    Args:
        file_path:    txt file path.
        min_set_size: Minimum set size, the default is 2, which means that sets containing only one node
                      will be regarded as isolated sets and removed.
    """

    raw_data = []
    with open(file_path, encoding='utf8') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines if not line.strip().startswith('#')]

    one_element = lines[0].rsplit("," ,)[-1]
    if '@' in one_element:
        s_name, s_vec = one_element.split("@")
        has_ele_name = len(s_name.strip()) > 0              # 集合是否有元素名
        feat_dim = len(s_vec.strip()[1:-1].split(' '))      # 集合元素特征向量维度
    else:
        feat_dim = 0
        has_ele_name = True

    element_set = set()   # 所有不重复集合元素

    # 获得不重复元素name
    for line in lines:
        if ":" in line:
            _, eles = [s.strip() for s in line.split(':', 1)]
        else:
            eles = line
        eles = [e.strip().split('@') for e in eles.split(',')]
        if len(eles) < min_set_size:  # 排除小于min_size的集合
            continue
        raw_data.append(eles)
        element_set.update([e[0] for e in eles])

    ele_dict = {e: n for n, e in enumerate(element_set)} if has_ele_name else {}
    if element_dict is not None:
        ele_dict = element_dict
    ele_dict_r = {n: e for e, n in ele_dict.items()} if has_ele_name else {}

    set_lst = []

    # 构造集合列表
    for set_data in raw_data:
        set_eles = []
        for ele_raw in set_data:
            ele_name = ele_raw[0]
            if feat_dim == 0:
                set_eles.append(ele_dict[ele_name])
            else:
                ele_vec = [float(n) for n in ele_raw[1].strip()[1:-1].split(' ')]
                if has_ele_name:
                    set_eles.append((ele_dict[ele_name], ele_vec))
                else:
                    set_eles.append(ele_vec)

        set_lst.append(set_eles)

    return set_lst, ele_dict_r


def create_pkl(txt_path, pkl_path, test_p, fake_fn, min_set_size=2, element_dict=None):
    """
    txt_path: 文本数据文件路径
    pkl_path: pikle文件的存储路径
    test_p: 用于测试的比例
    fake_fn: 用于生成假集合的函数
    min_set_size: 最小集合大小
    element_dict: 预先指定的元素字典
    """
    real_sets, ele_dict_r = load_from_txt(txt_path, min_set_size, element_dict)
    set_num = len(real_sets)
    test_idxs = np.arange(set_num)
    np.random.shuffle(test_idxs)
    test_num = int(set_num*test_p)
    test_idxs = test_idxs[:test_num]
    ele_dict = element_dict if element_dict is not None else {v:k for k,v in ele_dict_r.items()}
    fake_sets = fake_fn(real_sets, ele_dict, test_num, min_set_size)

    train_sets = []  # 训练集
    test_sets = []   # 测试集
    for i, s in enumerate(real_sets):
        if i in test_idxs:
            test_sets.append((s, 1))
        else:
            train_sets.append(s)
    for s in fake_sets:
        test_sets.append((s, 0))

    # 构造用于baseline的训练数据集
    fake_sets_baseline = fake_fn(real_sets, ele_dict, set_num-test_num, min_set_size)
    train_sets_baseline = []
    for s in train_sets:
        train_sets_baseline.append((s, 1))
    for s in fake_sets_baseline:
        train_sets_baseline.append((s, 0))

    return {
            'train_sets': train_sets,
            'test_sets': test_sets,
            'train_sets_baseline': train_sets_baseline,
            'ele_num': len(ele_dict),
            'max_set_size': max(max(len(s) for s in real_sets), max(len(s) for s in fake_sets))
            }


def create_pkls(txt_path, pkl_path, test_p, fake_fn, min_set_size=2, element_dict=None, repeat=1):
    data_lst = []
    for _ in range(repeat):
        data_lst.append(create_pkl(txt_path, pkl_path, test_p, fake_fn, min_set_size, element_dict))

    with open(pkl_path, 'wb') as f:
        pickle.dump(data_lst, f)


def create_fake_by_cross(set_data, ele_dict, fake_num, min_set_size):
    """
    Cross-exchange the two sets at random positions to get two new fake sets. 
    Make sure that the false set is not in the true set.
    """
    fake_sets = []
    id_pairs = []
    n = 0
    set_num = len(set_data)
    while n < fake_num:
        pair = rand.sample(range(set_num), 2)
        if (pair[0], pair[1]) in id_pairs or (pair[1], pair[0]) in id_pairs:
            continue
        set1, set2 = set_data[pair[0]], set_data[pair[1]]

        rand_ids1 = list(range(len(set1)))
        rand.shuffle(rand_ids1)
        rand_ids2 = list(range(len(set2)))
        rand.shuffle(rand_ids2)

        tmp_set1 = [set1[i] for i in rand_ids1]
        tmp_set2 = [set2[i] for i in rand_ids2]

        point1 = rand.randint(1, len(set1)-1)
        point2 = rand.randint(1, len(set2)-1)

        fake_set = tmp_set1[:point1] + tmp_set2[point2:]
        if len(fake_set) >= min_set_size:
            fake_sets.append(fake_set)
            n += 1

        fake_set = tmp_set2[:point2] + tmp_set1[point1:]
        if len(fake_set) >= min_set_size:
            fake_sets.append(fake_set)
            n += 1

    return fake_sets


def create_fake_dist_randomly(set_data, ele_dict, fake_num, min_set_size, replace_ratio=0.5):
    """
    Construct a false set by randomly selecting elements from the true set. 
    The occurrence probability of an element is consistent with its original distribution.
    """
    fake_sets = []
    candi_eles = list(chain(*set_data))
    datas = rand.sample(set_data, fake_num)
    for data in datas:
        f_data = rand.sample(data, int(len(data)*(1 - replace_ratio)))
        ele_names = {d[0] if isinstance(d, tuple) else d for d in f_data}
        while True:
            f_ele = rand.choice(candi_eles)
            f_n = f_ele[0] if isinstance(f_ele, tuple) else f_ele
            if f_n in ele_names:
                continue
            ele_names.add(f_n)
            f_data.append(f_ele)
            if len(f_data) == len(data):
                break
        fake_sets.append(f_data)
    return fake_sets


def create_fake_full_randomly(set_data, ele_dict, fake_num, min_set_size, replace_ratio=0.5):
    """
    Elements are chosen completely randomly to replace the original elements.
    """
    fake_sets = []
    if isinstance(set_data[0][0], tuple):
        candi_eles = list({d[0] for d in chain(*set_data)})
        candi_feats = [d[1] for d in chain(*set_data)]
    else:
        candi_eles = list({d for d in chain(*set_data)})
        candi_feats = None
    datas = rand.sample(set_data, fake_num)
    for data in datas:
        f_data = rand.sample(data, int(len(data)*(1 - replace_ratio)))
        while True:
            f_ele = rand.choice(candi_eles)
            f_feats = rand.choice(candi_feats) if candi_feats else None
            if f_ele in f_data:
                continue
            if f_feats:
                f_data.append((f_ele, f_feats))
            else:
                f_data.append(f_ele)
            if len(f_data) == len(data):
                break
        fake_sets.append(f_data)
    return fake_sets


def create_fake_triangle(set_data, ele_dict, fake_num, min_set_size):
    fake_sets = []
    datas = rand.sample(set_data, fake_num)
    for data in datas:
        (x1, y1), (x2, y2) = rand.sample(data, 2)
        while True:
            x3 = rand.random()
            y3 = (x3 - x1)/(x2 - x1)*(y2 - y1) + y1
            if 0 < y3 < 1:
                fake_sets.append([[x1, y1],[x2, y2],[x3, y3]])
                break
    return fake_sets


def create_fake_setmnist(set_data, ele_dict, fake_num, min_set_size, digit):
    t = 0.3
    data_dir = './original_data_process/SetMNIST'
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(img_size)])
    mnist = MNIST(data_dir, train=True, transform=transform, download=True)
    rand_idx = list(range(len(mnist)))
    rand.shuffle(rand_idx)

    fake_sets = []
    for i in rand_idx:
        data, d = mnist[i]
        if d == digit:
            continue
        data = data[0]
        data[data>=t]=1
        data[data<t]=0
        fake_sets.append(data.nonzero().numpy().tolist())
        if len(fake_sets) >= fake_num:
            break
    return fake_sets


def create_fake_setmnist_randomly(set_data, ele_dict, fake_num, min_set_size):
    sizes = [len(data) for data in set_data]
    fake_sets = []
    for _ in range(fake_num):
        size = rand.choice(sizes)
        data = np.random.randint(0, img_size, [size,2])
        fake_sets.append(data.tolist())
    return fake_sets


def sample_pointclouds(data, num):
    mesh = o3d.io.read_triangle_mesh(data)
    # 随机采样：采样概率与三角形面积相关
    # sample = mesh.sample_points_uniformly(number_of_points=num)
    # 均匀采样
    sample =  mesh.sample_points_poisson_disk(number_of_points=num, init_factor=5)
    return np.asarray(sample.points).tolist()


def create_fake_pointcloud(set_data, ele_dict, fake_num, min_set_size, data_name):
    data_paths = './original_data_process/ModelNet10/data_paths.pkl'
    with open(data_paths, 'rb') as f:
        data_paths = pickle.load(f)
    datas = []
    for k, v in data_paths.items():
        if k != data_name:
            datas.extend(v)
    idx = list(range(fake_num))
    rand.shuffle(idx)
    fake_data_paths = [f"./original_data_process/ModelNet10/{datas[i][2:]}" for i in idx]
    return [sample_pointclouds(p, 200) for p in fake_data_paths]


def gen_ele_dict(data_name, min_set_size=2):
    real_path = f"./txts/{data_name}.txt"
    fake_path = f"./txts/{data_name}_fake.txt"
    _, real_dict = load_from_txt(real_path, min_set_size)
    _, fake_dict = load_from_txt(fake_path, min_set_size)
    eles = list(real_dict.values()) + list(fake_dict.values())
    eles = list(set(eles))
    ele_dict = {k:v for v, k in enumerate(eles)}
    return ele_dict


def create_fake_metabolic_reactions(set_data, ele_dict, fake_num, min_set_size, data_name):
    fake_path = f'./txts/{data_name}_fake.txt'
    fake_sets, _ = load_from_txt(fake_path, min_set_size, ele_dict)
    idx = list(range(len(fake_sets)))
    rand.shuffle(idx)
    idx = idx[:fake_num]
    return [fake_sets[i] for i in idx]


if __name__ == "__main__":
      # Randomly generate 10 pieces of data. The metrics in the paper is the average of these 10 points of data.
    repeat = 10

    txt_path = './txts'
    pkl_path = './pkls'

    # Recipes, compounds, shopping basket datasets
    data_names = ['chuancai', 'yuecai',
        'market_basket_optimisation', 'market_basket', 'groceries',
        'inorganic_compound', 'inorganic_compound_with_atom_num']
    # data_names = ['inorganic_compound_with_atom_num']
    for data_name in data_names:
        # fake_fn = partial(create_fake_dist_randomly, replace_ratio=0.99)
        fake_fn = partial(create_fake_full_randomly, replace_ratio=0.5)
        create_pkls(f'{txt_path}/{data_name}.txt', f'{pkl_path}/{data_name}.pkl', test_p=0.2, fake_fn=fake_fn, repeat=repeat)
        # create_pkls(f'{txt_path}/{data_name}.txt', f'{pkl_path}/{data_name}.pkl', test_p=0.2, fake_fn=create_fake_by_cross, repeat=repeat)

    # triangle dataset
    data_name = 'triangle'
    create_pkls(f'{txt_path}/{data_name}.txt', f'{pkl_path}/{data_name}.pkl', test_p=0.2, fake_fn=create_fake_triangle, repeat=repeat)

    # 2D point cloud (set MNIST)
    img_size = 10
    data_names = [('setmnist1', 1), ('setmnist8', 8)]
    for data_name, digit in data_names:
        fake_fn = partial(create_fake_setmnist, digit=digit)
        create_pkls(f'{txt_path}/{data_name}.txt', f'{pkl_path}/{data_name}.pkl', test_p=0.2, fake_fn=fake_fn, repeat=repeat)

    # 2D point cloud (set MNIST) with completely random negative samples
    img_size = 10
    data_names = ['setmnist1', 'setmnist8']
    for data_name in data_names:
        create_pkls(f'{txt_path}/{data_name}.txt', f'{pkl_path}/{data_name}_rand.pkl', test_p=0.2, fake_fn=create_fake_setmnist_randomly, repeat=repeat)

    # 3D point cloud
    data_names = [('modelnet10_chair', 'chair'), ('modelnet10_sofa', 'sofa')]
    for data_name, data in data_names:
        fake_fn = partial(create_fake_pointcloud, data_name=data)
        create_pkls(f'{txt_path}/{data_name}.txt', f'{pkl_path}/{data_name}.pkl', test_p=0.2, fake_fn=fake_fn, repeat=repeat)

    # Metabolic reactions datasets
    data_names = ["iAB_RBC_283", "iAF692", "iAF1260b", "iHN637", "iIT341", "iJO1366"]
    for data_name in data_names:
        fake_fn = partial(create_fake_metabolic_reactions, data_name=data_name)
        ele_dict = gen_ele_dict(data_name)
        create_pkls(f'{txt_path}/{data_name}.txt', f'{pkl_path}/{data_name}.pkl', test_p=0.2, fake_fn=fake_fn, element_dict=ele_dict, repeat=repeat)

    print("OVER!")
