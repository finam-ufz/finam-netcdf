"""
Custom input classes to transform grids to reversed axes.
"""

import finam as fm


class _NetCDFInput(fm.sdk.Input):

    def __init__(
        self, name, info=None, static=False, force_axes_reversed=False, **info_kwargs
    ):
        super().__init__(name, info, static, **info_kwargs)
        self._force_axes_reversed = force_axes_reversed

    def exchange_info(self, info=None):
        """Exchange the data info with the input's source.

        Parameters
        ----------
        info : :class:`.Info`
            request parameters

        Returns
        -------
        dict
            delivered parameters
        """
        self.logger.trace("exchanging info")

        with fm.tools.log_helper.ErrorLogger(self.logger):
            if self._in_info_exchanged:
                raise fm.errors.FinamMetaDataError("Input info was already exchanged.")
            if self._input_info is not None and info is not None:
                raise fm.errors.FinamMetaDataError(
                    "An internal info was already provided"
                )
            if self._input_info is None and info is None:
                raise fm.errors.FinamMetaDataError("No metadata provided")
            if info is None:
                info = self._input_info

            if not isinstance(info, fm.Info):
                raise fm.errors.FinamMetaDataError("Metadata must be of type Info")

        src_info = self._source.get_info(info)

        with fm.tools.log_helper.ErrorLogger(self.logger):
            fail_info = {}
            if not info.accepts(src_info, fail_info):
                fail_info = "\n".join(
                    [
                        f"{name} - got {got}, expected {exp}"
                        for name, (got, exp) in fail_info.items()
                    ]
                )
                raise fm.errors.FinamMetaDataError(
                    f"Can't accept incoming data info. Failed entries:\n{fail_info}"
                )

        self._input_info = src_info.copy_with(
            use_none=False, time=info.time, grid=info.grid, mask=info.mask, **info.meta
        )

        if (
            self._force_axes_reversed
            and not isinstance(self._input_info, fm.UnstructuredGrid)
            and not self._input_info.grid.axes_reversed
        ):
            self._input_info.grid = _revert_axes(self._input_info.grid)

        self._in_info_exchanged = True
        with fm.tools.log_helper.ErrorLogger(self.logger):
            self._transform = src_info.grid.get_transform_to(self._input_info.grid)

        # pylint: disable-next=fixme
        # TODO: check if this is correct (was src_info before)
        return self._input_info


class _NetCDFCallbackInput(fm.sdk.CallbackInput):

    def __init__(
        self,
        callback,
        name,
        info=None,
        static=False,
        force_axes_reversed=False,
        **info_kwargs,
    ):
        super().__init__(callback, name, info, static, **info_kwargs)
        self._force_axes_reversed = force_axes_reversed

    def exchange_info(self, info=None):
        """Exchange the data info with the input's source.

        Parameters
        ----------
        info : :class:`.Info`
            request parameters

        Returns
        -------
        dict
            delivered parameters
        """
        self.logger.trace("exchanging info")

        with fm.tools.log_helper.ErrorLogger(self.logger):
            if self._in_info_exchanged:
                raise fm.errors.FinamMetaDataError("Input info was already exchanged.")
            if self._input_info is not None and info is not None:
                raise fm.errors.FinamMetaDataError(
                    "An internal info was already provided"
                )
            if self._input_info is None and info is None:
                raise fm.errors.FinamMetaDataError("No metadata provided")
            if info is None:
                info = self._input_info

            if not isinstance(info, fm.Info):
                raise fm.errors.FinamMetaDataError("Metadata must be of type Info")

        src_info = self._source.get_info(info)

        with fm.tools.log_helper.ErrorLogger(self.logger):
            fail_info = {}
            if not info.accepts(src_info, fail_info):
                fail_info = "\n".join(
                    [
                        f"{name} - got {got}, expected {exp}"
                        for name, (got, exp) in fail_info.items()
                    ]
                )
                raise fm.errors.FinamMetaDataError(
                    f"Can't accept incoming data info. Failed entries:\n{fail_info}"
                )

        self._input_info = src_info.copy_with(
            use_none=False, time=info.time, grid=info.grid, mask=info.mask, **info.meta
        )

        if (
            self._force_axes_reversed
            and not isinstance(self._input_info, fm.UnstructuredGrid)
            and not self._input_info.grid.axes_reversed
        ):
            self._input_info.grid = _revert_axes(self._input_info.grid)

        self._in_info_exchanged = True
        with fm.tools.log_helper.ErrorLogger(self.logger):
            self._transform = src_info.grid.get_transform_to(self._input_info.grid)

        # pylint: disable-next=fixme
        # TODO: check if this is correct (was src_info before)
        return self._input_info


def _revert_axes(grid):
    if isinstance(grid, fm.EsriGrid):
        return grid  # EsriGrid is always axes_reversed

    if isinstance(grid, fm.UniformGrid):
        return fm.UniformGrid(
            grid.dims,
            spacing=grid.spacing,
            origin=grid.origin,
            data_location=grid.data_location,
            order=grid.order,
            axes_reversed=True,
            axes_increase=grid.axes_increase,
            axes_attributes=grid.axes_attributes,
            axes_names=grid.axes_names,
            crs=grid.crs,
        )

    if isinstance(grid, fm.RectilinearGrid):
        return fm.RectilinearGrid(
            axes=grid.axes,
            data_location=grid.data_location,
            order=grid.order,
            axes_reversed=True,
            axes_attributes=grid.axes_attributes,
            axes_names=grid.axes_names,
            crs=grid.crs,
        )

    raise ValueError("unsupported grid class")  # unreachable
