import numpy as np 

class SingleGenerator(): 
    def __init__(self, range_):
        self.range_ = range_
    def sample(self, N): 
        return np.random.choice(self.range_, N)
    
class ConsecutiveGenerator(): 
    def __init__(self, depth_range, sizes, ascending=True, prescribed_depths=None):  
        self.depth_range = depth_range
        assert max(self.depth_range) <= len(sizes), 'depth > len(sizes)' 
        if ascending:
            self.sizes = np.sort(sizes)
        else: 
            self.sizes = np.sort(sizes)[::-1] 
        self.prescribed_depths = prescribed_depths 

    def generate_space(self, depth):
        return [self.sizes[i:i+depth] for i in range(len(self.sizes) - depth + 1)]
    
    def get_sampling_dict(self, sample_depths): 
        d = {} 
        for i in sample_depths: 
            if i not in d.keys(): 
                d[i] = 1 
            else: 
                d[i] += 1
        return d
    
    def sample(self, N=None): # N=None if prescribed
        if self.prescribed_depths is not None: 
            assert max(self.prescribed_depths) <= len(self.sizes), 'depth > len(sizes)'  
            sample_depths = self.prescribed_depths 
        else: 
            sample_depths = self.get_sampling_dict(np.random.choice(self.depth_range, N))
        
        acc = []
        for depth,n in sample_depths.items(): 
            if depth == 0: 
                for _ in range(n): acc.append(np.array([])) 
            else: 
                sample_space = np.array(self.generate_space(depth))
                for _ in range(n): 
                    acc.append(sample_space[np.random.choice(np.arange(len(sample_space)), 1)][0])  
        return acc

class MonotonicGenerator(ConsecutiveGenerator): 
    def __init__(self, depth_range, sizes, **kwargs):
        super().__init__(depth_range, sizes, **kwargs)
        
    def generate_space(self,depth):
        import itertools
        if depth == 0:
            return []
        
        acc = [[[i]] for i in list(self.sizes)]
        for _ in range(depth-1):
            temp = []
            for i in range(4):
                flatten = list(itertools.chain.from_iterable(acc[i:]))
                temp.append([[self.sizes[i]] + l for l in flatten])
            acc = temp
        return list(itertools.chain.from_iterable(acc))
    
class OneByOneGenerator(ConsecutiveGenerator): 
    def __init__(self, depth_range, sizes, protop_nums, **kwargs):
        super().__init__(np.array(depth_range)-1, sizes, **kwargs)
        self.protop_nums = protop_nums
    
    def sample(self, N=None): 
        samples = super().sample(N)
        protop_num = np.random.choice(self.protop_nums)
        return np.array([np.append(a.astype('int64'), [protop_num]) for a in samples ])
    
class HiddenAndKernelGenerator(): 
    def __init__(self, depth_range, hidden_sizes, kernel_sizes, ascending=(True, False)):
        self.depth_range = np.array(depth_range)
        self.hidden_sizes = hidden_sizes
        self.kernel_sizes = kernel_sizes
        self.ascending = ascending 
        
    def get_sampling_dict(self, sample_depths): 
        d = {} 
        for i in sample_depths: 
            if i not in d.keys(): 
                d[i] = 1 
            else: 
                d[i] += 1
        return d
    
    def sample(self, N=None):
        sample_depths = self.get_sampling_dict(np.random.choice(self.depth_range, N))
        hidden = ConsecutiveGenerator(self.depth_range, self.hidden_sizes, 
                                      ascending=self.ascending[0], prescribed_depths=sample_depths)
        kernel = MonotonicGenerator(self.depth_range, self.kernel_sizes, 
                                    ascending=self.ascending[1], prescribed_depths=sample_depths)
        return (hidden.sample(), kernel.sample())

class ParamGenerators(): 
    def __init__(self, generators_dict): 
        self.gen_dict = generators_dict 
        self.samples_dict = {}
        self.samples_list = []
        
    def sample(self, N):
        self.samples_dict = {k: v.sample(N) for k,v in self.gen_dict.items()}
        self.samples_dict['HIDDEN_SIZES'] = self.samples_dict['HIDDEN_AND_KERNEL_SIZES'][0]
        self.samples_dict['KERNEL_SIZES'] = self.samples_dict['HIDDEN_AND_KERNEL_SIZES'][1]
        del self.samples_dict['HIDDEN_AND_KERNEL_SIZES']
        
        for i in range(N): 
            self.samples_list.append({k: v[i] for k,v in self.samples_dict.items()}) 
            
        return self.samples_list 