"""
This script is adapted from arviz.inference_data https://python.arviz.org/en/stable/_modules/arviz/data/inference_data.html
and is a more generalized form of arviz's InferenceData object that can accept any type of of group. It is a data structure
for using netcdf groups with xarray.
"""

import importlib
import importlib.resources
import re
import uuid
from functools import lru_cache
from html import escape
from typing import Any
from typing import Iterator
from typing import List
from typing import Mapping
from typing import Optional
from typing import Tuple
from typing import Union

import xarray as xr
from arviz.data.base import _extend_xr_method

STATIC_FILES = ("static/html/icons-svg-inline.html", "static/css/style.css")


@lru_cache(None)
def _load_static_files():
    """Lazily load the resource files into memory the first time they are needed.

    Clone from xarray.core.formatted_html_template.
    """
    return [importlib.resources.files("arviz").joinpath(fname).read_text() for fname in STATIC_FILES]


class HtmlTemplate:
    """Contain html templates for GWInfernoData repr."""

    html_template = """
            <div>
              <div class='xr-header'>
                <div class="xr-obj-type">gwinferno.types.GWInfernoData</div>
              </div>
              <ul class="xr-sections group-sections">
              {}
              </ul>
            </div>
            """
    element_template = """
            <li class = "xr-section-item">
                  <input id="idata_{group_id}" class="xr-section-summary-in" type="checkbox">
                  <label for="idata_{group_id}" class = "xr-section-summary">{group}</label>
                  <div class="xr-section-inline-details"></div>
                  <div class="xr-section-details">
                      <ul id="xr-dataset-coord-list" class="xr-var-list">
                          <div style="padding-left:2rem;">{xr_data}<br></div>
                      </ul>
                  </div>
            </li>
            """
    _, css_style = _load_static_files()  # pylint: disable=protected-access
    specific_style = ".xr-wrap{width:700px!important;}"
    css_template = f"<style> {css_style}{specific_style} </style>"


def _compressible_dtype(dtype):
    """Check basic dtypes for automatic compression."""
    if dtype.kind == "V":
        return all(_compressible_dtype(item) for item, _ in dtype.fields.values())
    return dtype.kind in {"b", "i", "u", "f", "c", "S"}


class GWInfernoData(Mapping[str, xr.Dataset]):
    """Adapted from Arviz InferenceData object https://python.arviz.org/en/stable/_modules/arviz/data/inference_data.html#InferenceData"""

    def __init__(
        self,
        attrs: Union[None, Mapping[Any, Any]] = None,
        **kwargs: Union[xr.Dataset, List[xr.Dataset], Tuple[xr.Dataset, xr.Dataset]],
    ) -> None:

        self._groups: List[str] = []
        self._attrs: Union[None, dict] = dict(attrs) if attrs is not None else None
        key_list = [key for key in kwargs]
        for key in key_list:
            dataset = kwargs[key]
            setattr(self, key, dataset)
            self._groups.append(key)

    @property
    def _groups_all(self) -> List[str]:
        return self._groups

    def __len__(self) -> int:
        """Return the number of groups in this GWInfernoData object."""
        return len(self._groups)

    def __iter__(self) -> Iterator[str]:
        """Iterate over groups in GWInfernoData object."""
        for group in self._groups:
            yield group

    def __getitem__(self, key: str) -> xr.Dataset:
        """Get item by key."""
        if key not in self._groups:
            raise KeyError(key)
        return getattr(self, key)

    def groups(self) -> List[str]:
        """Return all groups present in GWInfernoData object."""
        return self._groups_all

    def __repr__(self) -> str:
        """Make string representation of GWInfernoData object."""
        msg = "Inference data with groups:\n\t> {options}".format(options="\n\t> ".join(self._groups))
        return msg

    def _repr_html_(self) -> str:
        """Make html representation of GWInfernoData object."""
        try:
            from xarray.core.options import OPTIONS

            display_style = OPTIONS["display_style"]
            if display_style == "text":
                html_repr = f"<pre>{escape(repr(self))}</pre>"
            else:
                elements = "".join(
                    [
                        HtmlTemplate.element_template.format(
                            group_id=group + str(uuid.uuid4()),
                            group=group,
                            xr_data=getattr(self, group)._repr_html_(),  # pylint: disable=protected-access
                        )
                        for group in self._groups_all
                    ]
                )
                formatted_html_template = HtmlTemplate.html_template.format(elements)  # pylint: disable=possibly-unused-variable
                css_template = HtmlTemplate.css_template  # pylint: disable=possibly-unused-variable
                html_repr = f"{locals()['formatted_html_template']}{locals()['css_template']}"
        except Exception:
            html_repr = f"<pre>{escape(repr(self))}</pre>"
        return html_repr

    def to_netcdf(
        self,
        filename: str,
        compress: bool = True,
        groups: Optional[List[str]] = None,
        engine: str = "h5netcdf",
    ) -> str:
        """Write GWInfernoData to netcdf4 file.

        Parameters
        ----------
        filename : str
            Location to write to
        compress : bool, optional
            Whether to compress result. Note this saves disk space, but may make
            saving and loading somewhat slower (default: True).
        groups : list, optional
            Write only these groups to netcdf file.
        engine : {"h5netcdf", "netcdf4"}, default "h5netcdf"
            Library used to read the netcdf file.

        Returns
        -------
        str
            Location of netcdf file
        """
        mode = "w"  # overwrite first, then append
        if self._attrs:
            xr.Dataset(attrs=self._attrs).to_netcdf(filename, mode=mode, engine=engine)
            mode = "a"

        if self._groups:  # check's whether a group is present or not.
            if groups is None:
                groups = self._groups
            else:
                groups = [group for group in self._groups_all if group in groups]

            for group in groups:
                data = getattr(self, group)
                kwargs = {"engine": engine}
                if compress:
                    kwargs["encoding"] = {
                        var_name: {"zlib": True} for var_name, values in data.variables.items() if _compressible_dtype(values.dtype)
                    }
                data.to_netcdf(filename, mode=mode, group=group, **kwargs)
                data.close()
                mode = "a"
        elif not self._attrs:  # creates a netcdf file for an empty GWInfernoData object.
            if engine == "h5netcdf":
                import h5netcdf

                empty_netcdf_file = h5netcdf.File(filename, mode="w")
            elif engine == "netcdf4":
                import netCDF4 as nc

                empty_netcdf_file = nc.Dataset(filename, mode="w", format="NETCDF4")
            empty_netcdf_file.close()
        return filename

    @staticmethod
    def from_netcdf(filename, *, engine="h5netcdf", group_kwargs=None, regex=False) -> "GWInfernoData":
        """Initialize object from a netcdf file.

        Expects that the file will have groups, each of which can be loaded by xarray.
        By default, the datasets of the GWInfernoData object will be lazily loaded instead
        of being loaded into memory. This
        behaviour is regulated by the value of ``az.rcParams["data.load"]``.

        Parameters
        ----------
        filename : str
            location of netcdf file
        engine : {"h5netcdf", "netcdf4"}, default "h5netcdf"
            Library used to read the netcdf file.
        group_kwargs : dict of {str: dict}, optional
            Keyword arguments to be passed into each call of :func:`xarray.open_dataset`.
            The keys of the higher level should be group names or regex matching group
            names, the inner dicts re passed to ``open_dataset``
            This feature is currently experimental.
        regex : bool, default False
            Specifies where regex search should be used to extend the keyword arguments.
            This feature is currently experimental.

        Returns
        -------
        GWInfernoData
        """
        groups = {}
        attrs = {}

        if engine == "h5netcdf":
            import h5netcdf
        elif engine == "netcdf4":
            import netCDF4 as nc
        else:
            raise ValueError(f"Invalid value for engine: {engine}. Valid options are: h5netcdf or netcdf4")

        try:
            with h5netcdf.File(filename, mode="r") if engine == "h5netcdf" else nc.Dataset(filename, mode="r") as data:
                data_groups = list(data.groups)

            for group in data_groups:
                group_kws = {}

                group_kws = {}
                if group_kwargs is not None and regex is False:
                    group_kws = group_kwargs.get(group, {})
                if group_kwargs is not None and regex is True:
                    for key, kws in group_kwargs.items():
                        if re.search(key, group):
                            group_kws = kws
                group_kws.setdefault("engine", engine)
                with xr.open_dataset(filename, group=group, **group_kws) as data:
                    groups[group] = data

            with xr.open_dataset(filename, engine=engine) as data:
                attrs.update(data.load().attrs)

            return GWInfernoData(attrs=attrs, **groups)
        except OSError as err:
            if err.errno == -101:
                raise type(err)(
                    str(err)
                    + (
                        " while reading a NetCDF file. This is probably an error in HDF5, "
                        "which happens because your OS does not support HDF5 file locking.  See "
                        "https://stackoverflow.com/questions/49317927/"
                        "errno-101-netcdf-hdf-error-when-opening-netcdf-file#49317928"
                        " for a possible solution."
                    )
                ) from err
            raise err

    def extend(self, other, join="left"):
        """Extend InferenceData with groups from another InferenceData.

        Parameters
        ----------
        other : InferenceData
            InferenceData to be added
        join : {'left', 'right'}, default 'left'
            Defines how the two decide which group to keep when the same group is
            present in both objects. 'left' will discard the group in ``other`` whereas 'right'
            will keep the group in ``other`` and discard the one in ``self``.

        Examples
        --------
        Take two InferenceData objects, and extend the first with the groups it doesn't have
        but are present in the 2nd InferenceData object.

        First InferenceData:

        .. jupyter-execute::

            import arviz as az
            idata = az.load_arviz_data("rugby")

        Second InferenceData:

        .. jupyter-execute::

            other_idata = az.load_arviz_data("radon")

        Call the ``extend`` method:

        .. jupyter-execute::

            idata.extend(other_idata)
            idata

        See how now the first InferenceData has more groups, with the data from the
        second one, but the groups it originally had have not been modified,
        even if also present in the second InferenceData.

        See Also
        --------
        add_groups : Add new groups to InferenceData object.
        concat : Concatenate InferenceData objects.

        """
        if not isinstance(other, GWInfernoData):
            raise ValueError("Extending is possible between two InferenceData objects only.")
        if join not in ("left", "right"):
            raise ValueError(f"join must be either 'left' or 'right', found {join}")
        for group in other._groups_all:  # pylint: disable=protected-access
            if hasattr(self, group) and join == "left":
                continue
            dataset = getattr(other, group)
            setattr(self, group, dataset)
            self._groups.append(group)

    set_index = _extend_xr_method(xr.Dataset.set_index, see_also="reset_index")
    get_index = _extend_xr_method(xr.Dataset.get_index)
    reset_index = _extend_xr_method(xr.Dataset.reset_index, see_also="set_index")
    set_coords = _extend_xr_method(xr.Dataset.set_coords, see_also="reset_coords")
    reset_coords = _extend_xr_method(xr.Dataset.reset_coords, see_also="set_coords")
    assign = _extend_xr_method(xr.Dataset.assign)
    assign_coords = _extend_xr_method(xr.Dataset.assign_coords)
    sortby = _extend_xr_method(xr.Dataset.sortby)
    chunk = _extend_xr_method(xr.Dataset.chunk)
    unify_chunks = _extend_xr_method(xr.Dataset.unify_chunks)
    load = _extend_xr_method(xr.Dataset.load)
    compute = _extend_xr_method(xr.Dataset.compute)
    persist = _extend_xr_method(xr.Dataset.persist)
    quantile = _extend_xr_method(xr.Dataset.quantile)

    # The following lines use methods on xr.Dataset that are dynamically defined and attached.
    # As a result mypy cannot see them, so we have to suppress the resulting mypy errors.
    mean = _extend_xr_method(xr.Dataset.mean, see_also="median")  # type: ignore[attr-defined]
    median = _extend_xr_method(xr.Dataset.median, see_also="mean")  # type: ignore[attr-defined]
    min = _extend_xr_method(xr.Dataset.min, see_also=["max", "sum"])  # type: ignore[attr-defined]
    max = _extend_xr_method(xr.Dataset.max, see_also=["min", "sum"])  # type: ignore[attr-defined]
    cumsum = _extend_xr_method(xr.Dataset.cumsum, see_also="sum")  # type: ignore[attr-defined]
    sum = _extend_xr_method(xr.Dataset.sum, see_also="cumsum")  # type: ignore[attr-defined]
