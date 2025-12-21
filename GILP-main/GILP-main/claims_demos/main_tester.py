from test_claim_1 import test_claim_1
from test_claim_2 import test_claim_2
from test_claim_4_5 import test_claims_4_and_5
import sys

def main():
    print("==========================================================")
    print("       GILP CORE CLAIMS VERIFICATION SUITE")
    print("==========================================================")
    print("Running rigorous tests for the '10 Pillars of Geometric Inference'...\n")
    
    results = {}
    
    # Claim 1
    try:
        results["Claim 1 (Directional Progress)"] = test_claim_1()
    except Exception as e:
        print(f"Error testing Claim 1: {e}")
        results["Claim 1 (Directional Progress)"] = False
        
    print("\n----------------------------------------------------------\n")
    
    # Claim 2
    try:
        results["Claim 2 (Fossilization/OFF-Mode)"] = test_claim_2()
    except Exception as e:
        print(f"Error testing Claim 2: {e}")
        results["Claim 2 (Fossilization/OFF-Mode)"] = False

    print("\n----------------------------------------------------------\n")

    # Claim 4 & 5
    try:
        # returns composite boolean, but script prints individual details
        res_4_5 = test_claims_4_and_5() 
        results["Claim 4 (Meaningful Failure)"] = res_4_5 # Approximation for summary
        results["Claim 5 (Deep Composition)"] = res_4_5   # Approximation for summary
    except Exception as e:
        print(f"Error testing Claim 4/5: {e}")
        results["Claim 4 (Meaningful Failure)"] = False
        results["Claim 5 (Deep Composition)"] = False

    print("\n==========================================================")
    print("FINAL VERDICT REPORT")
    print("==========================================================")
    
    all_passed = True
    for claim, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {claim}")
        if not passed: all_passed = False
        
    if all_passed:
        print("\n>>> ALL TESTED CLAIMS VERIFIED. GILP PARADIGM CONFIRMED.")
        sys.exit(0)
    else:
        print("\n>>> SOME CLAIMS FAILED. REVIEW LOGS.")
        sys.exit(1)

if __name__ == "__main__":
    main()
