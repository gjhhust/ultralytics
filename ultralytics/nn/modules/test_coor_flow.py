import torch
import unittest
from ultralytics.nn.modules.flow  import AlternateCorrBlock 

class TestAlternateCorrBlock(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.height = 16
        self.width = 16
        self.channels = 64
        self.num_levels = 3
        self.radius = 3
        self.stride = [1, 1, 1]
        self.fmap1 = torch.randn(self.batch_size, self.channels, self.height, self.width).cuda().half()
        self.fmaps2 = [torch.randn(self.batch_size, self.channels, self.height // (2 ** i), self.width // (2 ** i)).cuda().half() for i in range(self.num_levels)]
        self.coords = torch.randn(self.batch_size, 2, self.height, self.width).cuda().half()

    def test_alternate_corr_block_initialization(self):
        corr_block = AlternateCorrBlock(self.fmap1, self.fmaps2, self.num_levels, self.radius, self.stride)
        self.assertEqual(len(corr_block.pyramid), self.num_levels)
        for i in range(self.num_levels):
            self.assertEqual(corr_block.pyramid[i][0].shape, self.fmap1.shape)
            self.assertEqual(corr_block.pyramid[i][1].shape, self.fmaps2[i].shape)

    def test_alternate_corr_block_forward_pass(self):
        corr_block = AlternateCorrBlock(self.fmap1, self.fmaps2, self.num_levels, self.radius, self.stride)
        output = corr_block(self.coords)
        self.assertEqual(output.shape, (self.batch_size, self.num_levels * (2 * self.radius + 1) ** 2, self.height, self.width))

    def test_alternate_corr_block_edge_case_single_level(self):
        corr_block = AlternateCorrBlock(self.fmap1, self.fmaps2[:1], num_levels=1, radius=self.radius, stride=self.stride)
        output = corr_block(self.coords)
        self.assertEqual(output.shape, (self.batch_size, (2 * self.radius + 1) ** 2, self.height, self.width))

    def test_alternate_corr_block_edge_case_zero_radius(self):
        corr_block = AlternateCorrBlock(self.fmap1, self.fmaps2, self.num_levels, radius=0, stride=self.stride)
        output = corr_block(self.coords)
        self.assertEqual(output.shape, (self.batch_size, self.num_levels, self.height, self.width))

    def test_alternate_corr_block_edge_case_zero_stride(self):
        corr_block = AlternateCorrBlock(self.fmap1, self.fmaps2, self.num_levels, self.radius, stride=[0, 0, 0])
        output = corr_block(self.coords)
        self.assertEqual(output.shape, (self.batch_size, self.num_levels * (2 * self.radius + 1) ** 2, self.height, self.width))

if __name__ == '__main__':
    unittest.main()