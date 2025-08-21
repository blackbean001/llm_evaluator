import TopsAnalytics as ta
import time
import os
import argparse

CAPE = False
if CAPE:
    device_name = 'zixiao_v2x2'
else:
    device_name = 'scorpiox2'

def find_all_vpd(path):
    vpd_list = []
    with os.scandir(path) as entries:
        for entry in entries:
            if entry.is_file():
                if entry.name[-4:] == ".vpd":
                  vpd_list.append(entry.path)
            elif entry.is_dir():
                vpd_list += find_all_vpd(entry.path)
    return vpd_list


def process_vpd_files(file_list, report_dir, run_mode):
    start = time.time()
    error_list = []
    for f in file_list:
        #    f2 = report_dir + "/" + f.replace("/", "_").replace(".vpd", ".xlsx")
        f2 = report_dir + "/" + f.split("/")[-2] + ".xlsx"
        if os.path.exists(f2):
            print(f"skip exist file: {f}")
            continue
        print("NOW PROCESSING: ", f)
        try:
            opinfo_collector = ta.parse_information_from_vpd(
                f,
                method="correlation",
                time_stamp_high="inf",  # 1739091932455357023,
                time_stamp_low=0,
                host_name=None,  # ["sse-jq-116-252"],
                device_id=None,  # [1,2,3,4,5],
                stream_id=None,  # [0,1,2,3,4,5,6,7],
                verbose=True,
                max_process_number=100000,
                run_mode=run_mode)
        except:
            error_list.append(f)
            continue

        df = ta.export_dataframe(
            opinfo_collector,
            {
                "index": ta.IndexVisitor(),
                "start_timestamp": ta.StartTimestampVisitor(),
                "end_timestamp": ta.EndTimestampVisitor(),
                "op_id": ta.OpIdVisitor(),
                "gcu_op": ta.OpNameVisitor(),
                "topsflame_op": ta.TraceNameVisitor(),
                "input_shape": ta.InputShapeVisitor(),
                "output_shape": ta.OutputShapeVisitor(),
                "attribute": ta.AttributeVisitor(),
                "1d_floats": ta.ElementWiseFloatsVisitor(),
                "1d_sfu": ta.SFUFloatsVisitor(),
                "2d_floats": ta.TensorCoreFloatsVisitor(),
                "input_size": ta.InputSizeVisitor(),
                "output_size": ta.OutputSizeVisitor(),
                "total_size": ta.TotalSizeVisitor(),
                "SFU_floats": ta.SFUFloatsVisitor(),
                "1D_flops_ratio": ta.ElementWiseTFlopsRatioVisitor(device_name),
                "2D_flops_ratio": ta.TensorCoreTFlopsRatioVisitor(device_name),
                "SFU_flops_ratio": ta.SFUTFlopsRatioVisitor(device_name),
                "busy(ns)": ta.DurationCycleVisitor(),
                "gap(ns)": ta.GapsVisitor(),
                "topsflameop_handle": ta.TopsflameOpHandleVisitor(),
                "top_topsflameop_handle": ta.TopTopsflameOpHandleVisitor(),
                "top_topsflameop_handle_name": ta.TopTopsFlameOpHandleNameVisitor(),
                "top_topsflameop_start_timestamp": ta.TopTopsFlameOpStartTimestampVisitor(),
                "top_topsflameop_end_timestamp": ta.TopTopsFlameOpEndTimestampVisitor(),
                "topsflameop_start_timestamp": ta.TopsFlameOpStartTimestampVisitor(),
                "topsflameop_end_timestamp": ta.TopsFlameOpEndTimestampVisitor(),
                "topsflameop_duration": ta.TopsFlameOpDurationTimestampVisitor(),
                "launch_kernel_to_op_run_gap": ta.LaunchKernelToOpRunGapVisitor(),
                "runtime_start_timestamp":ta.RuntimeStartTimestampVisitor(),
                "runtime_end_timestamp":ta.RuntimeEndTimestampVisitor(),
                "runtime_duration_timestamp":ta.RuntimeDurationTimestampVisitor(),
                "target": ta.ExtremePerformanceVisitor(device_name),
                "bound": ta.BoundTypeVisitor(device_name),
                "device_id": ta.DeviceIdVisitor(),
                "stream_id": ta.StreamIdVisitor(),
                "host_name": ta.HostNameVisitor(),
                "step": ta.StepVisitor(method ="graph"),
            },
            saved_name=f2,
        )

    if len(error_list) > 0:
        print("files faild to process:")
        for file in error_list:
            print(file)

    print("done!")
    end = time.time()
    print("time cost True: ", end - start)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", type=str,
                        help='folder where profiles store in')
    parser.add_argument("--report", type=str,
                        help='folder where TopsAnalytics report store in')
    parser.add_argument("--mode", type=str,
                        choices=["1c4s", "1c12s", "2c24s"], default="1c4s")
    args = parser.parse_args()

    file_list = find_all_vpd(args.profiles)
    if not os.path.exists(args.report):
        os.mkdir(args.report)
    process_vpd_files(file_list, args.report, args.mode)
