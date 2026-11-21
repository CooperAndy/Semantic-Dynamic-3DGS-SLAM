// SPDX-License-Identifier: GPL-3.0-or-later
// Standalone observer of the existing ORB-SLAM3 stereo API; no core changes.
#include <System.h>
#include <Map.h>
#include <MapPoint.h>
#include <Eigen/Dense>
#include <opencv2/imgcodecs.hpp>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <thread>

int main(int argc,char**argv) {
    if(argc!=6){std::cerr<<"vocabulary settings sequence output source_first\n";return 1;}
    const std::string sequence=argv[3],output=argv[4];
    const int source_first=std::stoi(argv[5]);
    std::ifstream times(sequence+"/times.txt");std::vector<double> stamps;double stamp;
    while(times>>stamp)stamps.push_back(stamp);
    if(stamps.size()!=1000){std::cerr<<"Expected exactly1000 timestamps\n";return 2;}
    for(size_t i=1;i<stamps.size();++i)if(!(stamps[i]>stamps[i-1]))return 2;
    std::ofstream ledger(output+"/causal_ledger.csv"),raw(output+"/raw_c2w_NOT_FOR_MAPPING.txt");
    if(!ledger||!raw)return 2;
    ledger<<"frame,source_frame,timestamp,tracking_state,valid_pose,map_changed,map_id,stereo_matches,original_left,kept_left,original_right,kept_right,elapsed_ms,max_input_index";
    for(int r=0;r<4;++r)for(int c=0;c<4;++c)ledger<<",tcw_"<<r<<c;
    ledger<<std::endl;raw<<std::setprecision(17);
    ORB_SLAM3::System slam(argv[1],argv[2],ORB_SLAM3::System::STEREO,false);
    if(std::abs(slam.GetImageScale()-1.f)>1e-7){slam.Shutdown();return 2;}
    int valid_count=0;std::set<long> map_ids;
    for(int i=0;i<1000;++i){
        std::ostringstream name;name<<std::setw(6)<<std::setfill('0')<<i;
        // Only this iteration's pair is decoded; no future images are passed.
        cv::Mat left=cv::imread(sequence+"/image_2/"+name.str()+".png",cv::IMREAD_UNCHANGED);
        cv::Mat right=cv::imread(sequence+"/image_3/"+name.str()+".png",cv::IMREAD_UNCHANGED);
        if(left.empty()||right.empty()||left.size()!=right.size()){slam.Shutdown();return 2;}
        ORB_SLAM3::TsddMaskStats stats;
        const auto begin=std::chrono::steady_clock::now();
        Sophus::SE3f pose=slam.TrackStereo(left,right,stamps[i],std::vector<ORB_SLAM3::IMU::Point>(),name.str(),cv::Mat(),ORB_SLAM3::TsddMaskPolicy::O0_NONE,&stats);
        const auto returned=std::chrono::steady_clock::now();
        const Eigen::Matrix4d tcw=pose.matrix().cast<double>();
        const int state=slam.GetTrackingState();const bool changed=slam.MapChanged();long map_id=-1;
        for(auto* point:slam.GetTrackedMapPoints())if(point&&!point->isBad()){
            ORB_SLAM3::Map* map=point->GetMap();if(map){map_id=static_cast<long>(map->GetId());break;}
        }
        const Eigen::Matrix3d rotation=tcw.block<3,3>(0,0);
        const bool valid=state==2&&tcw.allFinite()&&std::abs(rotation.determinant()-1.)<1e-3&&(rotation.transpose()*rotation-Eigen::Matrix3d::Identity()).norm()<1e-3&&map_id>=0;
        Eigen::Matrix4d twc=Eigen::Matrix4d::Constant(std::numeric_limits<double>::quiet_NaN());
        if(valid){twc.setIdentity();const Eigen::Matrix3d inverse=rotation.inverse();twc.block<3,3>(0,0)=inverse;twc.block<3,1>(0,3)=-inverse*tcw.block<3,1>(0,3);++valid_count;map_ids.insert(map_id);}
        std::ostringstream row;row<<std::setprecision(17)<<i<<','<<source_first+i<<','<<stamps[i]<<','<<state<<','<<int(valid)<<','<<int(changed)<<','<<map_id<<','<<slam.GetLastStereoMatches()<<','<<stats.original_left<<','<<stats.kept_left<<','<<stats.original_right<<','<<stats.kept_right<<','<<std::chrono::duration<double,std::milli>(returned-begin).count()<<','<<i;
        for(int r=0;r<4;++r)for(int c=0;c<4;++c)row<<','<<tcw(r,c);
        row<<'\n';ledger<<row.str()<<std::flush;
        for(int r=0;r<3;++r)for(int c=0;c<4;++c)raw<<twc(r,c)<<((r==2&&c==3)?'\n':' ');
        raw.flush();
        // Pace like the source stream after sealing this returned pose.
        const double dt=i<999?stamps[i+1]-stamps[i]:stamps[i]-stamps[i-1];
        const double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
        if(elapsed<dt)std::this_thread::sleep_for(std::chrono::duration<double>(dt-elapsed));
    }
    ledger.close();raw.close();slam.Shutdown();
    slam.SaveTrajectoryKITTI(output+"/offline_final_NOT_FOR_MAPPING.txt");
    std::ofstream summary(output+"/runner_summary.txt");summary<<"frames 1000\nvalid_poses "<<valid_count<<"\nunique_map_ids "<<map_ids.size()<<"\n";
    return 0;
}
