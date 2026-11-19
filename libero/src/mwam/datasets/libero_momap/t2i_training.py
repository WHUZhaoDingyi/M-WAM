"""Shared LIBERO episode access and task foreground selection."""
from collections import OrderedDict
import os

import h5py
from torch.utils.data import Dataset
from mwam.robotwin_momap import RawEpisodeReader, MomapTargetService

def foreground_ids(registry,objects):
    matched=set();ids=[]
    for entry in registry:
        name=entry['entity_name'];hits=[o for o in objects if name==o or name.startswith(o+'_')];matched.update(hits)
        if hits or name.startswith(('robot0_','gripper0_')):ids.append(int(entry['raw_seg_id']))
    if set(objects)-matched:raise ValueError('Missing task objects in current entity registry')
    return ids


class LiberoT2IDataset(Dataset):
    def __len__(self):return len(self.indices) if self.indices is not None else (self.end[-1] if self.end else 0)
    def close(self):
        for r,s,h in self.cache.values():r.close();s.close();h.close()
        self.cache.clear()
    def __getstate__(self):
        state=self.__dict__.copy();state['cache']=OrderedDict();state['pid']=None;return state
    def _open(self,path):
        if self.pid!=os.getpid():self.close();self.pid=os.getpid()
        if path not in self.cache:
            self.cache[path]=(RawEpisodeReader(path),MomapTargetService(path),h5py.File(path,'r',swmr=True,libver='latest'))
            while len(self.cache)>self.max_open:
                _,(r,s,h)=self.cache.popitem(last=False);r.close();s.close();h.close()
        self.cache.move_to_end(path);return self.cache[path]
    def __getitem__(self,index):return self.load_window(index,include_text=True)
