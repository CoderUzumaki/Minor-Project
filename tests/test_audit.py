import importlib.util
from pathlib import Path
import numpy as np
from pyquaternion import Quaternion

spec=importlib.util.spec_from_file_location('audit',Path(__file__).parents[1]/'scripts/audit_radar_overlap.py')
audit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_causal_selection_never_uses_future():
    assert audit.latest_causal([90,110],100,20)==0
    assert audit.latest_causal([110],100,20) is None
    assert audit.latest_causal([70],100,20) is None
    assert audit.latest_causal([100],100,0)==0


def test_sensor_to_global_rotation_translation():
    calibration={'rotation':Quaternion(axis=[0,0,1],angle=np.pi/2).elements,'translation':[1,0,0]}
    pose={'rotation':[1,0,0,0],'translation':[10,20,0]}
    result=audit.transform_points(np.array([[2.],[0.],[0.]]),calibration,pose)
    np.testing.assert_allclose(result[:,0],[11,22,0],atol=1e-10)


def test_box_dimensions_rotation_and_ambiguity():
    box={'translation':[0,0,0],'rotation':Quaternion(axis=[0,0,1],angle=np.pi/2).elements,'size':[2,4,2]}
    pts=np.array([[0,1.5,0],[1.5,0,2.2],[0,0,0]])
    hit=audit.box_membership(pts,[box],0)
    assert hit.tolist()==[[True,False,False]]
    assert audit.box_membership(pts,[box],.5).tolist()==[[True,True,True]]
    counts,ambiguous=audit.exclusive_counts(np.array([[True,True,False],[True,False,True]]))
    assert counts.tolist()==[1,1] and ambiguous==1


def test_episodes_require_fixed_pair_and_consecutive_frames():
    def row(i,sensors):
        return dict(scene='s',instance_token='a',sensors=sensors,frame_index=i,timestamp_us=i*500000,sample_token=str(i),category='vehicle.car')
    rows=[row(0,['A','B']),row(1,['A','C']),row(2,['A','B']),row(3,['A','B'])]
    result=audit.sustained_episodes(rows,2)
    assert max(r['keyframes'] for r in result)==2
    assert result[0]['sensors']=='A|B'


def test_archive_path_traversal_rejected():
    import pytest
    prep_spec=importlib.util.spec_from_file_location('prep',Path(__file__).parents[1]/'scripts/prepare_nuscenes.py')
    prep=importlib.util.module_from_spec(prep_spec)
    prep_spec.loader.exec_module(prep)
    with pytest.raises(ValueError):
        prep.selected_member('../escape')
    assert prep.selected_member('samples/RADAR_FRONT/example.pcd')
    assert not prep.selected_member('samples/CAM_FRONT/example.jpg')
