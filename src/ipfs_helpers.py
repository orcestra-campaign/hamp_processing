import numcodecs
import xarray as xr
import fsspec
import io


def get_chunks(dimensions):
    match dimensions:
        case ("time",):
            chunks = {
                "time": 2**18,
            }
        case ("time", "height"):
            chunks = {
                "time": 2**11,
                "height": 2**7,
            }
        case ("time", "frequency"):
            chunks = {
                "time": 2**16,
                "frequency": 5,
            }

    return tuple((chunks[d] for d in dimensions))


def get_encoding(dataset):
    numcodecs.blosc.set_nthreads(1)  # IMPORTANT FOR DETERMINISTIC CIDs
    codec = numcodecs.Blosc("zstd", shuffle=1, clevel=6)

    return {
        var: {
            "chunks": get_chunks(dataset[var].dims),
            "compressor": codec,
        }
        for var in dataset.variables
        if var not in dataset.dims
    }


def read_nc(url):
    with fsspec.open(url, "rb", expand=True) as fp:
        bio = io.BytesIO(fp.read())
        return xr.open_dataset(bio, engine="scipy")


def read_mf_nc(url):
    ds = xr.open_mfdataset(
        fsspec.open_local(f"simplecache::{url}"),
        combine_attrs="drop_conflicts",
    )
    return ds


async def get_client(**kwargs):
    import aiohttp

    conn = aiohttp.TCPConnector(limit=1)
    return aiohttp.ClientSession(connector=conn, **kwargs)
