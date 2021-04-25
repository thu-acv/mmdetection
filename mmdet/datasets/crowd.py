import os
import os.path as osp
import xml.etree.ElementTree as ET

import mmcv
import pickle
from PIL import Image
from .builder import DATASETS
from .xml_style import XMLDataset


@DATASETS.register_module()
class CrowdDataset(XMLDataset):

    CLASSES = ('person',)

    def __init__(self, **kwargs):
        super(CrowdDataset, self).__init__(**kwargs)


    def load_annotations(self, ann_file):
        """Load annotation from XML style ann_file.

        Args:
            ann_file (str): Path of XML file.

        Returns:
            list[dict]: Annotation info from XML file.
        """
        
        data_infos = []
        img_ids = mmcv.list_from_file(ann_file)

        for img_id in img_ids:
            filename = f'JPEGImages/{img_id}.jpg'
            xml_path = osp.join(self.img_prefix, 'Annotations',
                                f'{img_id}.xml')
            size = None
            if not self.has_no_annotation:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                size = root.find('size')
            if size is not None:
                width = int(size.find('width').text)
                height = int(size.find('height').text)
            else:
                img_path = osp.join(self.img_prefix, 'JPEGImages',
                                    '{}.jpg'.format(img_id))
                if not os.path.exists(img_path):
                    width, height = 0, 0
                else:
                    img = Image.open(img_path)
                    width, height = img.size
            data_infos.append(
                dict(id=img_id, filename=filename, width=width, height=height))

        return data_infos

    def format_results(self, results, **kwargs):
        """Format the results to pickle file.

        Args:
            results (list[tuple | numpy.ndarray]): Testing results of the
                dataset.

        Returns:
            None
        """
        
        with open('crowd_results.pkl', 'wb') as f:
            pickle.dump(results, f)
        