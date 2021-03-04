
import random
from typing import Iterator, Sequence

from torch.utils.data import Sampler
from torch.utils.data.sampler import T_co


# noinspection PyMissingConstructor
class EqualSampler(Sampler):

    def __init__(self, idxs_per_class: Sequence[Sequence[float]], replacement: bool = True) -> None:

        self.replacement = replacement
        self.idxs_per_class = idxs_per_class
        self.len_largest_class = max([len(x) for x in self.idxs_per_class])

    def __iter__(self) -> Iterator[T_co]:
        return iter(random.choice(self.idxs_per_class[random.randint(0, len(self.idxs_per_class)-1)])
                    for _ in range(len(self)))

    def __len__(self):
        return self.len_largest_class * len(self.idxs_per_class)


if __name__ == '__main__':
    es = EqualSampler([list(range(5)), list(range(5, 10)), list(range(10, 12))])
    for i in es:
        print(i)
    pass
