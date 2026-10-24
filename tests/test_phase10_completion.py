from datetime import datetime,timedelta,timezone
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from src.completion_replay import viewing_union,hourly_parts,scheduled_parts,weekly_replay,conditional_completion,covered,exact_context

KYIV=ZoneInfo("Europe/Kyiv")


def utc(y,m,d,h,minute=0,fold=0):
    return datetime(y,m,d,h,minute,tzinfo=KYIV,fold=fold).astimezone(timezone.utc)


def test_exact_hourly_allocation_unions_overlap_and_preserves_midnight_and_dst():
    a=utc(2020,5,1,19,30);b=utc(2020,5,1,21,30)
    views=pl.DataFrame({"user_id":["u","u"],"channel_id":[1,1],"view_start_utc":[a,a+timedelta(minutes=30)],"view_end_utc":[b,b]})
    parts=hourly_parts(viewing_union(views,a,b)).sort("part_start")
    assert parts["hour"].to_list()==[19,20,21]
    assert parts["seconds"].to_list()==[1800.,3600.,1800.]
    for start,end,hours,seconds in ((utc(2020,5,1,23,30),utc(2020,5,2,1,30),[23,0,1],7200),
        (utc(2020,3,29,2,30),utc(2020,3,29,5,30),[2,4,5],7200),
        (utc(2020,10,25,2,30),utc(2020,10,25,4,30),[2,3,3,4],10800)):
        f=pl.DataFrame({"a":[start],"b":[end]})
        p=hourly_parts(f).sort("part_start")
        assert p["hour"].to_list()==hours
        assert p["seconds"].sum()==seconds
        if hours==[23,0,1]:assert p["dow"][0]!=p["dow"][-1]


def test_broadcast_weights_are_fractional_airtime_not_start_slot():
    a=utc(2020,5,1,19,30);b=utc(2020,5,1,21,30)
    schedule=pl.DataFrame({"airing_id":["a"],"tv_show_id":[1],"channel_id":[1],
        "programme_start_utc":[a],"programme_end_utc":[b],"programme_duration_seconds":[7200]})
    f=scheduled_parts(schedule,a,b).sort("part_start")
    assert f["weight"].to_list()==[.25,.5,.25]
    assert f["weight"].sum()==1


def test_weekly_wraparound_matches_sunday_and_following_monday_without_duplicates(tmp_path):
    start=utc(2020,3,9,0);cutoff=utc(2020,3,23,0)
    a=utc(2020,3,15,23,30);b=utc(2020,3,16,0,30)
    views=pl.DataFrame({"user_id":["u","u"],"channel_id":[1,1],
        "view_start_utc":[a,a+timedelta(minutes=10)],"view_end_utc":[b,b]})
    union=viewing_union(views,start,cutoff)
    fa=utc(2020,3,29,23,30);fb=utc(2020,3,30,0,30)
    schedule=pl.DataFrame({"airing_id":["future"],"tv_show_id":[1],"channel_id":[1],
        "programme_start_utc":[fa],"programme_end_utc":[fb],"programme_duration_seconds":[3600]})
    future=scheduled_parts(schedule,cutoff,fb)
    result=weekly_replay(union,future,pl.DataFrame({"user_id":["u"],"tv_show_id":[1]}),start,cutoff,tmp_path/"weekly.parquet")
    assert result["weekly50"].to_list()==[1.]
    assert result["weekly80"].to_list()==[1.]
    assert 0<result["weekly_decay"][0]<1


def test_weekly_coverage_integral_is_exact_for_disjoint_and_touching_intervals():
    a=np.array([0.,10.,20.]);b=np.array([5.,20.,30.]);w=np.array([1.,.5,.25])
    profile=(a,b,np.r_[0,np.cumsum(b-a)],w,np.r_[0,np.cumsum((b-a)*w)])
    assert np.allclose(covered(profile,np.array([-1.,3.,7.,15.,35.])),[0.,3.,5.,10.,25.])
    assert np.isclose(covered(profile,np.array([30.]),True)[0],12.5)


def test_conditional_probabilities_use_channel_prior_and_keep_partial_interest():
    start=utc(2020,3,9,0);cutoff=utc(2020,3,16,0);end=utc(2020,3,23,0)
    schedule=pl.DataFrame({"airing_id":["past","future"],"tv_show_id":[1,2],"channel_id":[1,1],
        "programme_start_utc":[utc(2020,3,10,20),utc(2020,3,17,20)],
        "programme_end_utc":[utc(2020,3,10,21),utc(2020,3,17,21)],"programme_duration_seconds":[3600,3600]})
    airings=pl.DataFrame({"user_id":["u"],"airing_id":["past"],"channel_id":[1],"tv_show_id":[1],"coverage":[.6],"seconds":[2160]})
    frame,names,behavior=conditional_completion(pl.DataFrame({"user_id":["u"],"tv_show_id":[2]}),airings,
        scheduled_parts(schedule,start,cutoff),scheduled_parts(schedule,cutoff,end),["u"],{2},extra=0)
    assert len(names)==8
    assert frame["completion50_channel"][0]>frame["completion80_channel"][0]>0
    # channel80 prior=(0+2*(1/3))/(1+2)=2/9, smoothed P=(0+5*2/9)/6=5/27.
    assert np.isclose(frame["completion80_channel"][0],(5/27)/11)
    assert behavior["meaningful_broadcasts"].to_list()==[1]
    assert behavior["completed_broadcasts"].to_list()==[0]


def test_hourly_cutoff_does_not_count_future_seconds_or_subsecond_loss():
    a=utc(2020,5,1,19,30)+timedelta(microseconds=250000);cutoff=utc(2020,5,1,20)
    views=pl.DataFrame({"user_id":["u"],"channel_id":[1],"view_start_utc":[a],"view_end_utc":[cutoff+timedelta(hours=2)]})
    f=hourly_parts(viewing_union(views,a,cutoff))
    assert np.isclose(f["seconds"].sum(),1799.75)
    assert f["part_end"].max()==cutoff


def test_exact_context_uses_all_viewed_hours_in_all_recent_profiles():
    a=utc(2020,5,1,19,30);b=utc(2020,5,1,21,30);cutoff=utc(2020,5,2,0)
    views=pl.DataFrame({"user_id":["u"],"channel_id":[1],"view_start_utc":[a],"view_end_utc":[b]})
    schedule=pl.DataFrame({"airing_id":["f1","f2","f3"],"tv_show_id":[1,2,3],"channel_id":[1]*3,
        "programme_start_utc":[utc(2020,5,8,h) for h in (19,20,21)],
        "programme_end_utc":[utc(2020,5,8,h) for h in (20,21,22)],"programme_duration_seconds":[3600]*3})
    frame=pl.DataFrame({"user_id":["u"]*3,"tv_show_id":[1,2,3]})
    result,_=exact_context(frame,viewing_union(views,a,cutoff),scheduled_parts(schedule,cutoff,utc(2020,5,9,0)),cutoff)
    for days in (7,14,28):
        assert np.allclose(result[f"compatibility_channel_hour_{days}"].to_numpy(),[.25,.5,.25])
        assert np.allclose(result[f"compatibility_channel_dow_hour_{days}"].to_numpy(),[.25,.5,.25])


def test_batched_week_queries_equal_individual_exact_integrals():
    a=np.array([0.,10.,604800.+3.,604800.+20.]);b=a+5;w=np.array([1.,.9,.8,.7])
    profile=(a,b,np.r_[0,np.cumsum(b-a)],w,np.r_[0,np.cumsum((b-a)*w)])
    endpoints=np.array([[3.,13.,25.],[604800.+5.,604800.+23.,604800.+30.]])
    for weighted in (False,True):
        assert np.allclose(covered(profile,endpoints,weighted),np.vstack([covered(profile,row,weighted) for row in endpoints]))
