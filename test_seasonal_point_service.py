import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import xarray as xr

from services import seasonal_point_service as service


class SeasonalPointTest(unittest.TestCase):
    def write(self, root, values=None, unit='kg m-2 s-1'):
        path=Path(root)/'CFSv2.prate.202611.ENSMEAN.fcst.nc'
        values=np.arange(3*3*4,dtype=float).reshape(3,3,4)/86400 if values is None else values
        times=['2026-12-01','2027-01-01','2027-02-01']
        xr.Dataset({'fcst':(('time','lat','lon'),values,{'units':unit})},coords={'time':np.array(times,dtype='datetime64[ns]'),'lat':[31.,32.,33.],'lon':[117.,118.,119.,120.]}).to_netcdf(path,engine='scipy')
        meta={'file_name':path.name,'variable_layers':{'fcst':{'varType':'precipitation','unit':'mm/day','period_months':1,'frames':[{'valid_time':t+'T00:00:00Z'} for t in times]}}}
        return path,meta,[t+'T00:00:00Z' for t in times]

    def test_values_from_original_point_exact_months_units_and_orientation(self):
        with tempfile.TemporaryDirectory() as root:
            path,meta,times=self.write(root)
            result=service.sample_series(path,meta,'fcst',times,118.8,31.925)
            self.assertEqual(result['sample_point'],{'longitude':119.,'latitude':32.})
            np.testing.assert_allclose([v['value'] for v in result['series']],[6.,18.,30.])
            self.assertEqual(result['unit'],'mm/day')
            self.assertEqual(result['grid_spacing'],{'longitude':1.,'latitude':1.})
            self.assertEqual(len(result['neighbors']),9)
            self.assertFalse(result['regional_average'])
            self.assertEqual(result['interpolation'],'none')
            self.assertEqual(result['series'][0]['valid_time'], '2026-12-01T00:00:00+00:00')

    def test_missing_values_not_zero_and_requested_order_is_preserved(self):
        with tempfile.TemporaryDirectory() as root:
            values=np.ones((3,3,4));values[1,1,2]=np.nan
            path,meta,times=self.write(root,values,'mm/day')
            result=service.sample_series(path,meta,'fcst',[times[2],times[1],times[0]],118.8,31.925)
            self.assertEqual([r['value'] for r in result['series']],[1.,None,1.])
            json.dumps(result,allow_nan=False)

    def test_region_grid_uses_all_inside_centers_and_preserves_orientation_units_and_missing(self):
        with tempfile.TemporaryDirectory() as root:
            path,meta,times=self.write(root)
            result=service.sample_series(path,meta,'fcst',times,0,0,bbox=[117.5,31.5,119.5,32.5])
            self.assertEqual(result['longitudes'],[118.,119.])
            self.assertEqual(result['latitudes'],[32.])
            np.testing.assert_allclose(result['series'][0]['values'],[[5.,6.]])
            np.testing.assert_allclose(result['series'][2]['values'],[[29.,30.]])
            self.assertEqual(result['unit'],'mm/day')
            self.assertEqual(result['method'],'original_region_grid')
            empty=service.sample_series(path,meta,'fcst',times,0,0,bbox=[117.1,31.1,117.2,31.2])
            self.assertEqual(empty['status'],'no_data')
            self.assertEqual(empty['longitudes'],[])
            with self.assertRaises(service.PointSeriesError):service.sample_series(path,meta,'fcst',times,0,0,bbox=[100,0,120,33])

    def test_wrong_time_variable_unit_and_outside_domain_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path,meta,times=self.write(root)
            for variable,selected,lon,lat in [('missing',times,119,32),('fcst',['2030-01-01T00:00:00Z'],119,32),('fcst',times,0,0),('fcst',[times[0],times[0]],119,32)]:
                with self.assertRaises(service.PointSeriesError):service.sample_series(path,meta,variable,selected,lon,lat)
            meta['variable_layers']['fcst']['unit']='mm'
            with self.assertRaises(service.PointSeriesError):service.sample_series(path,meta,'fcst',times,119,32)

    def test_probability_encoding_and_season_window_preserved(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'prate.202611.prob.seas.nc'
            xr.Dataset({k:(('time','lat','lon'),np.full((1,2,2),v),{'units':'percent'}) for k,v in [('prob_above',.6),('prob_norm',.3),('prob_below',.1)]},coords={'time':np.array(['2026-12-01'],dtype='datetime64[ns]'),'lat':[31.,32.],'lon':[118.,119.]}).to_netcdf(path,engine='scipy')
            meta={'file_name':path.name,'variable_layers':{'prob_above':{'unit':'%','varType':'probability','frames':[{'valid_time':'2026-12-01T00:00:00Z','forecast_label':'2026-12 至 2027-02'}]}}}
            result=service.sample_series(path,meta,'prob_above',['2026-12-01T00:00:00Z'],118.8,31.925)
            self.assertEqual(result['series'][0]['value'],60.)
            self.assertEqual(result['period_months'],3)

    def test_grib_original_grid_uses_same_adapter_units_and_frame_identity(self):
        from eccodes import codes_grib_new_from_samples,codes_release,codes_set,codes_set_values,codes_write
        with tempfile.TemporaryDirectory() as root:
            source=Path(root)/'seasonal_tp.grib2'
            handle=codes_grib_new_from_samples('regular_ll_sfc_grib2')
            try:
                for key,value in [('Ni',4),('Nj',3),('latitudeOfFirstGridPointInDegrees',40.),('latitudeOfLastGridPointInDegrees',20.),('longitudeOfFirstGridPointInDegrees',100.),('longitudeOfLastGridPointInDegrees',130.),('iDirectionIncrementInDegrees',10.),('jDirectionIncrementInDegrees',10.),('shortName','tp'),('stepType','accum'),('dataDate',20260601),('dataTime',0),('step',3)]:
                    codes_set(handle,key,value)
                codes_set_values(handle,np.arange(12,dtype=float)/1000)
                with source.open('wb') as output:codes_write(handle,output)
            finally:codes_release(handle)
            meta=service.adapter.process_file(str(source))
            time=meta['variable_layers']['tp']['frames'][0]['valid_time']
            result=service.sample_series(source,meta,'tp',[time],119.,31.9)
            self.assertEqual(result['sample_point'],{'longitude':120.,'latitude':30.})
            self.assertAlmostEqual(result['series'][0]['value'],6.,places=4)
            self.assertEqual(result['unit'],'mm')

    def test_auth_and_resource_scope_checked_before_any_file_or_cache_access(self):
        user={'uuid':'u','role':2,'token_version':4}
        with patch.object(service,'get_active_user',return_value=user),patch.object(service,'get_display_resource',return_value=None) as access,patch.object(service,'sample_series') as read:
            for token,status in [({'sub':'u','token_version':3},401),({'sub':'u','token_version':4},404)]:
                with self.assertRaises(service.PointSeriesError) as caught:service.authorized_series(token,'file','fcst',['2026-12-01T00:00:00Z'],119,32)
                self.assertEqual(caught.exception.status,status)
            self.assertEqual(access.call_args.args[1:3],('u',2))
            read.assert_not_called()

    def test_database_path_escape_rejected(self):
        user={'uuid':'u','role':2,'token_version':4}
        resource={'data_type':'SEASONAL_PRECIP','source_path':'../../private.nc','meta_path':'../../private.json'}
        with patch.object(service,'get_active_user',return_value=user),patch.object(service,'get_display_resource',return_value=resource):
            with self.assertRaises(service.PointSeriesError):service.authorized_series({'sub':'u','token_version':4},'file','fcst',['2026-12-01T00:00:00Z'],119,32)


if __name__=='__main__':unittest.main()
