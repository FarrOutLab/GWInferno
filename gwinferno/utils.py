import xarray as xr


from typing import (
    Any,
    List,
    Iterator,
    Mapping,
    Tuple,
    Union,
    Optional,
    overload,
)


def _compressible_dtype(dtype):
    """Check basic dtypes for automatic compression."""
    if dtype.kind == "V":
        return all(_compressible_dtype(item) for item, _ in dtype.fields.values())
    return dtype.kind in {"b", "i", "u", "f", "c", "S"}


class DataSet(Mapping[str, xr.Dataset]):
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

    def __len__(self) -> int:
        """Return the number of groups in this InferenceData object."""
        return len(self._groups)
    
    def __iter__(self) -> Iterator[str]:
        """Iterate over groups in InferenceData object."""
        for group in self._groups:
            yield group

    def __getitem__(self, key: str) -> xr.Dataset:
        """Get item by key."""
        if key not in self._groups:
            raise KeyError(key)
        return getattr(self, key)
        
    def to_netcdf(
        self,
        filename: str,
        compress: bool = True,
        groups: Optional[List[str]] = None,
        engine: str = "h5netcdf",
    ) -> str:
        """Write InferenceData to netcdf4 file.

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
                        var_name: {"zlib": True}
                        for var_name, values in data.variables.items()
                        if _compressible_dtype(values.dtype)
                    }
                data.to_netcdf(filename, mode=mode, group=group, **kwargs)
                data.close()
                mode = "a"
        elif not self._attrs:  # creates a netcdf file for an empty InferenceData object.
            if engine == "h5netcdf":
                import h5netcdf

                empty_netcdf_file = h5netcdf.File(filename, mode="w")
            elif engine == "netcdf4":
                import netCDF4 as nc

                empty_netcdf_file = nc.Dataset(filename, mode="w", format="NETCDF4")
            empty_netcdf_file.close()
        return filename
    
    @staticmethod
    def from_netcdf(
        filename, *, engine="h5netcdf", group_kwargs=None, regex=False
    ) -> "InferenceData":
        """Initialize object from a netcdf file.

        Expects that the file will have groups, each of which can be loaded by xarray.
        By default, the datasets of the InferenceData object will be lazily loaded instead
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
        InferenceData
        """
        groups = {}
        attrs = {}

        if engine == "h5netcdf":
            import h5netcdf
        elif engine == "netcdf4":
            import netCDF4 as nc
        else:
            raise ValueError(
                f"Invalid value for engine: {engine}. Valid options are: h5netcdf or netcdf4"
            )

        try:
            with h5netcdf.File(filename, mode="r") if engine == "h5netcdf" else nc.Dataset(
                filename, mode="r"
            ) as data:
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

            return DataSet(attrs=attrs, **groups)
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