#!/usr/bin/env python3
"""
硬件连通性测试脚本 - 只读取状态，不发送控制指令
用法:
    python scripts/test_hardware.py --test cameras
    python scripts/test_hardware.py --test arm --port can_right
    python scripts/test_hardware.py --test all --port can_right
"""

import argparse
import time


def test_orbbec_cameras():
    """检测并列出所有 Orbbec 相机"""
    print("\n" + "="*50)
    print("  ORBBEC CAMERA DETECTION")
    print("="*50)
    
    try:
        import pyorbbecsdk as ob
        ctx = ob.Context()
        device_list = ctx.query_devices()
        count = len(device_list) if hasattr(device_list, "__len__") else device_list.get_count()
        
        if count == 0:
            print("⚠️  No Orbbec cameras found!")
            return False
        
        print(f"✓ Found {count} Orbbec camera(s):\n")
        cameras = []
        for i in range(count):
            dev = device_list.get_device_by_index(i)
            dev_info = dev.get_device_info()
            name = dev_info.get_name()
            serial = dev_info.get_serial_number()
            cameras.append({"index": i, "name": name, "serial": serial})
            print(f"  [{i}] Name: {name}")
            print(f"      Serial Number: {serial}")
            print(f"      Use in config: \"index_or_path\": \"{serial}\"")
            print()
        
        return True
    except Exception as e:
        print(f"✗ Orbbec camera detection failed: {e}")
        return False


def test_piper_arm(port: str = "can_right", skip_bitrate_check: bool = True):
    """测试 Piper 机械臂连通性（只读取状态，不发送控制指令）"""
    print("\n" + "="*50)
    print("  PIPER ARM CONNECTION TEST")
    print("="*50)
    print(f"  CAN Port: {port}")
    print("="*50 + "\n")
    
    try:
        from piper_sdk import C_PiperInterface
        
        print(f"[1/4] Creating PiperInterface on {port}...")
        piper = C_PiperInterface(can_name=port)
        print("      ✓ Interface created")
        
        print(f"[2/4] Connecting to CAN port...")
        piper.ConnectPort()
        print("      ✓ CAN port connected")
        
        # 等待数据稳定
        time.sleep(0.3)
        
        print(f"[3/4] Reading joint states (READ-ONLY, no motor control)...")
        joint_msgs = piper.GetArmJointMsgs()
        gripper_msgs = piper.GetArmGripperMsgs()
        
        print("\n  Current Joint Positions:")
        print("  -------------------------")
        for i in range(1, 7):
            val = getattr(joint_msgs.joint_state, f"joint_{i}")
            print(f"    Joint {i}: {val:>10}")
        
        gripper_val = gripper_msgs.gripper_state.grippers_angle
        print(f"    Gripper: {gripper_val:>10}")
        
        print(f"\n[4/4] Disconnecting...")
        piper.DisconnectPort()
        print("      ✓ Disconnected")
        
        print("\n" + "="*50)
        print("  ✓ PIPER ARM CONNECTION TEST PASSED!")
        print("="*50)
        return True
        
    except Exception as e:
        print(f"\n✗ Piper arm test failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Check if arm is powered on")
        print("  2. Check CAN cable connection")
        print("  3. Verify CAN interface is UP: ip link show can_right")
        print("  4. Check bitrate: ip -details link show can_right")
        print("  5. Install can-utils and test: candump can_right")
        return False


def test_opencv_cameras():
    """检测 OpenCV 可用的摄像头"""
    print("\n" + "="*50)
    print("  OPENCV CAMERA DETECTION")
    print("="*50)
    
    try:
        import cv2
        found = []
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    h, w = frame.shape[:2]
                    found.append({"index": i, "resolution": f"{w}x{h}"})
                cap.release()
        
        if not found:
            print("⚠️  No OpenCV cameras found!")
            return False
        
        print(f"✓ Found {len(found)} OpenCV camera(s):\n")
        for cam in found:
            print(f"  [/dev/video{cam['index']}] Resolution: {cam['resolution']}")
            print(f"      Use in config: \"index_or_path\": {cam['index']}")
        return True
    except Exception as e:
        print(f"✗ OpenCV camera detection failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Hardware connectivity test")
    parser.add_argument("--test", choices=["cameras", "arm", "all", "orbbec", "opencv"], 
                        default="all", help="What to test")
    parser.add_argument("--port", type=str, default="can_right", 
                        help="CAN port for arm (default: can_right)")
    args = parser.parse_args()
    
    results = {}
    
    if args.test in ["cameras", "all", "orbbec"]:
        results["orbbec"] = test_orbbec_cameras()
    
    if args.test in ["cameras", "all", "opencv"]:
        results["opencv"] = test_opencv_cameras()
    
    if args.test in ["arm", "all"]:
        results["arm"] = test_piper_arm(args.port)
    
    # 总结
    print("\n" + "="*50)
    print("  SUMMARY")
    print("="*50)
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name:15}: {status}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
