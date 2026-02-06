from pytorch_lightning.cli import LightningCLI, LightningArgumentParser
import tempfile
import sys

import torch
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
from numpy.core import multiarray
from numpy.dtypes import Float64DType
from numpy import ndarray, dtype

if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage, multiarray._reconstruct, ndarray, dtype, Float64DType])

class MyCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_argument("--test_only", action='store_true', default=None)
        parser.add_argument("-n", "--dry_run", action='store_true')
        parser.add_argument("--ckpt_path", type=str, required=False, default=None)


def run_training(cli: MyCLI) -> None:
    model = cli.model
    # Train and test, or test only
    if not cli.config.test_only:
        cli.trainer.fit(model, cli.datamodule, ckpt_path=cli.config.ckpt_path)
        cli.trainer.test(model, datamodule=cli.datamodule, ckpt_path=cli.config.ckpt_path)
        print("Test only complete.")
        # cli.trainer.save_checkpoint(cli.config.ckpt_path)
    else:
        cli.trainer.test(model, datamodule=cli.datamodule, ckpt_path=cli.config.ckpt_path)

def main(arguments=None):
    cli = MyCLI(run=False)

    if cli.config.dry_run:
        with tempfile.TemporaryDirectory() as tempdir:
            print("Dry run selected, outputting to {}".format(tempdir))
            cli = MyCLI(args=sys.argv[1:] + ['--trainer.default_root_dir', tempdir], run=False)
            run_training(cli)
    else:
        run_training(cli)


if __name__ == '__main__':
    main()
