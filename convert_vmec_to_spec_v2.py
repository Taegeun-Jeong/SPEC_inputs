#!/usr/bin/env python3
"""
This script converts VMEC input and output files into a SPEC input file.

It performs the following main steps:
1.  Reads VMEC wout file:
    - Extracts iota, pressure, and toroidal flux (phi) profiles.
    - Extracts the total toroidal flux (phiedge).
2.  Reads VMEC input file:
    - Extracts boundary Fourier coefficients (RBC, ZBS, etc.).
    - Extracts axis Fourier coefficients (RAXIS_CC, ZAXIS_CS).
    - Extracts the number of field periods (NFP).
3.  Generates a SPEC input file:
    - Divides the total toroidal flux into a user-specified number of shells.
    - Populates the &physicslist namelist with stepped pressure profiles based
      on these shells, interpolating values from the VMEC profiles.
    - Includes axis and boundary information in SPEC format.
    - Appends predefined &numericlist, &locallist, &globallist,
      &diagnosticslist, and &screenlist namelists with improved convergence parameters.
"""
import sys
import numpy as np
from netCDF4 import Dataset
import math

def extract_vmec_boundary(vmec_input_filename):
    """Extracts the boundary parameters from the VMEC input file more robustly."""
    with open(vmec_input_filename, 'r') as f:
        lines = f.readlines()
    
    boundary_lines = []
    in_boundary_namelist = False

    for line in lines:
        stripped = line.strip()
        line_upper = stripped.upper()
        
        # Start condition: Find the beginning of a namelist that contains boundary info.
        if '&' in stripped and ('RBC(' in line_upper or 'ZBS(' in line_upper):
            in_boundary_namelist = True

        if not in_boundary_namelist:
            continue
        
        # Add the line if it's not a comment
        if not stripped.startswith('!'):
            boundary_lines.append(stripped)

        # End condition: Find the namelist terminator
        if stripped.startswith('/'):
            break
            
    # Fallback for old format without clear namelist tags
    if not boundary_lines:
        inside = False
        for line in lines:
            line_upper = line.strip().upper()
            if "!---- BOUNDARY PARAMETERS" in line_upper:
                inside = True
                continue
            if inside:
                if line.strip().startswith("!"): continue
                if line.strip().startswith("/"):
                    boundary_lines.append("/")
                    break
                if line.strip().startswith("&"): break
                boundary_lines.append(line.strip())
        if boundary_lines and not boundary_lines[0].upper().startswith('RBC') and not boundary_lines[0].upper().startswith('ZBS'):
             boundary_lines = [] # False positive
        
    if not boundary_lines:
        print("Error: Could not robustly extract boundary parameters.")
        return []
        
    return boundary_lines

def wrap_boundary_block(boundary_lines):
    """Processes the boundary lines for SPEC format."""
    processed = []
    # SPEC boundary block is not a namelist, just values ending with '/'
    # The first line should not be a namelist tag like '&boundarylist'
    if boundary_lines[0].strip().startswith('&'):
        # Skip the namelist line itself
        lines_to_process = boundary_lines[1:]
    else:
        lines_to_process = boundary_lines

    for line in lines_to_process:
        # Strip comments
        if "!" in line:
            line = line.split("!")[0].strip()
        if line:
            processed.append(line)
    
    # Ensure it ends with a slash, and only one
    while processed and not processed[-1].strip():
        processed.pop()
    if not processed or processed[-1].strip() != "/":
        processed.append("/")
    elif len(processed) > 1 and processed[-2].strip() == "/":
        processed.pop() # Remove duplicate slash
        
    return processed

def extract_axis_parameters(vmec_input_filename):
    """Extracts axis parameters from the VMEC input file."""
    with open(vmec_input_filename, 'r') as f: lines = f.readlines()
    raxis_cc, zaxis_cs, raxis_cs_vmec, zaxis_cc_vmec = None, None, None, None
    inside_axis = False
    for line in lines:
        if "!---- AXIS PARAMETERS" in line.upper():
            inside_axis = True
            continue
        if inside_axis:
            stripped = line.strip()
            if not stripped or stripped.startswith("!"):
                if raxis_cc is not None and zaxis_cs is not None: break
                continue
            parts = stripped.split("=")
            if len(parts) >= 2:
                var_name = parts[0].strip().upper()
                coeffs_str = parts[1].split("!")[0].strip()
                try: coeffs = [float(c.replace('D','E')) for c in coeffs_str.split()]
                except ValueError: continue
                if var_name == "RAXIS_CC": raxis_cc = coeffs
                elif var_name == "ZAXIS_CS": zaxis_cs = coeffs
                elif var_name == "RAXIS_CS": raxis_cs_vmec = coeffs
                elif var_name == "ZAXIS_CC": zaxis_cc_vmec = coeffs
            if (raxis_cc and zaxis_cs and raxis_cs_vmec and zaxis_cc_vmec) or \
               stripped.startswith("!----") or stripped.startswith("&"):
                break
    return raxis_cc, zaxis_cs, raxis_cs_vmec, zaxis_cc_vmec

def format_spec_axis(raxis_cc, zaxis_cs, raxis_cs_vmec=None, zaxis_cc_vmec=None, ncoeff=10):
    """Formats axis parameters in SPEC style."""
    Rac = (raxis_cc[:ncoeff] + [0.0]*ncoeff)[:ncoeff] if raxis_cc else [0.0]*ncoeff
    Zas = (zaxis_cs[:ncoeff] + [0.0]*ncoeff)[:ncoeff] if zaxis_cs else [0.0]*ncoeff
    Ras = (raxis_cs_vmec[:ncoeff] + [0.0]*ncoeff)[:ncoeff] if raxis_cs_vmec else [0.0]*ncoeff
    Zac = (zaxis_cc_vmec[:ncoeff] + [0.0]*ncoeff)[:ncoeff] if zaxis_cc_vmec else [0.0]*ncoeff

    def format_line(label, values):
        return f"{label:<12}= {'  '.join(f'{v: .15E}' for v in values)}"

    return [format_line(" Rac", Rac), format_line(" Zas", Zas),
            format_line(" Ras", Ras), format_line(" Zac", Zac)]

def extract_vmec_scalar(vmec_input_filename, var_name, is_int=True):
    """Extracts a scalar value like NFP, MPOL, NTOR, CURTOR from VMEC input."""
    with open(vmec_input_filename, 'r') as f:
        for line in f:
            s_line = line.strip().upper()
            if not s_line.startswith("!") and var_name in s_line and "=" in s_line:
                parts = s_line.split("=")
                if len(parts) > 1:
                    val_str = parts[1].split("!")[0].strip().split()[0]
                    try:
                        val = float(val_str.replace('D', 'E'))
                        return int(val) if is_int else val
                    except (ValueError, IndexError): pass
    print(f"Warning: {var_name} not found in VMEC input file. Defaulting.")
    return 1 if var_name == "NFP" else 0

def format_array_for_spec(label, values, n_per_line=6, indent=12):
    """Formats a list of numbers into SPEC array style."""
    lines = []
    prefix = f"{label:<{indent}}="
    is_int_array = all(isinstance(v, (int, np.integer)) for v in values)
    
    for i in range(0, len(values), n_per_line):
        chunk = values[i:i+n_per_line]
        if is_int_array:
             line_values = "  ".join(map(str, chunk))
        else:
             line_values = "  ".join(f"{v: .15E}" for v in chunk)
        
        line_prefix = prefix if i == 0 else ' ' * (indent + 1)
        lines.append(f"{line_prefix} {line_values}")
    return "\n".join(lines)

def create_spec_physicslist_uniform_flux(wout_data, nfp_vmec, mpol_vmec, ntor_vmec, curtor_vmec, num_shells):
    """Creates the &physicslist namelist for SPEC, including the closing '/'."""
    phi_vmec, presf_vmec, iotaf_vmec = wout_data['phi'], wout_data['presf'], wout_data['iotaf']
    phiedge_vmec = wout_data['phiedge']
    
    Nvol_spec = num_shells
    flux_boundaries = np.linspace(0, phiedge_vmec, Nvol_spec + 1)
    tflux_spec = flux_boundaries[1:]
    
    pressure_at_boundaries = np.interp(flux_boundaries, phi_vmec, presf_vmec)
    iota_at_boundaries = np.interp(flux_boundaries, phi_vmec, iotaf_vmec)
    
    pressure_spec = (pressure_at_boundaries[:-1] + pressure_at_boundaries[1:]) / 2.0
    
    # --- MODIFIED PART ---
    # Set the 'adiabatic' array to be identical to the 'pressure' array.
    adiabatic_spec = np.copy(pressure_spec)
    # --- END OF MODIFIED PART ---
    
    iota_spec = iota_at_boundaries[:]
    oita_spec = iota_spec.copy()

    content = ["&physicslist",
               f" Igeometry   =         3", f" Istellsym   =         1", f" Lfreebound  =         0",
               f" phiedge     =  {phiedge_vmec: .15E}", f" curtor      =  {curtor_vmec: .15E}",
               f" curpol      =   0.000000000000000E+00", f" gamma       =   0.000000000000000E+00",
               f" Nfp         =         {nfp_vmec}", f" Nvol        =   {Nvol_spec}",
               f" Mpol        =         {mpol_vmec}", f" Ntor        =         {ntor_vmec}"]

    content.append(format_array_for_spec(" Lrad ", [12] * Nvol_spec))
    content.append(format_array_for_spec(" tflux", tflux_spec))
    content.append(format_array_for_spec(" helicity", [0.0] * Nvol_spec))
    content.append(f" pscale      =         1.0")
    content.append(f" Ladiabatic  =         2")
    content.append(format_array_for_spec(" pressure", pressure_spec))
    content.append(format_array_for_spec(" adiabatic", adiabatic_spec)) # Using the copied array
    content.append(format_array_for_spec(" iota", iota_spec))
    content.append(format_array_for_spec(" oita", oita_spec))
    content.append(f" Lconstraint =         0")
    content.append(f" mupftol     =   1.000000000000000E-14")
    content.append(f" mupfits     =         8")
    # content.append("/") # Namelist terminator

    return "\n".join(content)

def main_converter(vmec_input_file, wout_file, output_spec_file, num_shells):
    """Main function to orchestrate the conversion."""
    print(f"--- Starting VMEC to SPEC Conversion ---")
    print(f"  VMEC Input:  {vmec_input_file}")
    print(f"  Wout File:   {wout_file}")
    print(f"  SPEC Output: {output_spec_file}")
    print(f"  Num Shells:  {num_shells}")
    print(f"----------------------------------------")

    # 1. Read VMEC wout file
    print("Step 1: Reading VMEC wout file...")
    try:
        with Dataset(wout_file, 'r') as wout_fh:
            wout_content = {}
            vmec_vars_to_read = ['iotaf', 'presf', 'phi', 'ns', 'nfp', 'phiedge', 
                                 'raxis_cc', 'zaxis_cs']
            for var in vmec_vars_to_read:
                if var in wout_fh.variables:
                    wout_content[var] = wout_fh.variables[var][:]
                else:
                    if var == 'phiedge' and 'phi' in wout_fh.variables:
                        wout_content['phiedge'] = wout_fh.variables['phi'][-1]
                    else:
                        wout_content[var] = None
            if wout_content.get('nfp') is not None:
                wout_content['nfp'] = int(np.array(wout_content['nfp']).item())
    except Exception as e:
        print(f"Error: Could not read wout file '{wout_file}'.\n{e}"); sys.exit(1)

    # 2. Read VMEC input file
    print("Step 2: Reading VMEC input file for parameters...")
    nfp_in = extract_vmec_scalar(vmec_input_file, "NFP", is_int=True)
    mpol_in = extract_vmec_scalar(vmec_input_file, "MPOL", is_int=True)
    ntor_in = extract_vmec_scalar(vmec_input_file, "NTOR", is_int=True)
    curtor_in = extract_vmec_scalar(vmec_input_file, "CURTOR", is_int=False)

    if wout_content.get('nfp') and nfp_in != wout_content['nfp']:
        print(f"Warning: NFP mismatch (input: {nfp_in}, wout: {wout_content['nfp']}). Using NFP from input file.")

    raxis_cc_in, zaxis_cs_in, raxis_cs_in, zaxis_cc_in = extract_axis_parameters(vmec_input_file)
    if raxis_cc_in is None:
        raxis_cc_in = wout_content.get('raxis_cc')
        print("Using RAXIS_CC from wout file.")
    if zaxis_cs_in is None:
        zaxis_cs_in = wout_content.get('zaxis_cs')
        print("Using ZAXIS_CS from wout file.")
    if raxis_cc_in is None or zaxis_cs_in is None:
        print("Error: RAXIS/ZAXIS could not be found in input or wout file."); sys.exit(1)
            
    spec_axis_lines = format_spec_axis(raxis_cc_in, zaxis_cs_in, raxis_cs_in, zaxis_cc_in, ncoeff=10)

    vmec_boundary_raw = extract_vmec_boundary(vmec_input_file)
    if not vmec_boundary_raw: 
        print("Error: Could not extract boundary parameters from VMEC input."); sys.exit(1)
    spec_boundary_block_lines = wrap_boundary_block(vmec_boundary_raw)

    # 3. Generate SPEC file content
    print(f"Step 3: Generating SPEC content with {num_shells} shells...")
    physicslist_str = create_spec_physicslist_uniform_flux(
        wout_content, nfp_in, mpol_in, ntor_in, curtor_in, num_shells)

    # Assemble final file content
    spec_content = [physicslist_str, ""] + spec_axis_lines + [""] + spec_boundary_block_lines + [""]
    
    spec_content.append("""&numericlist
 Linitialize =         1
 Ndiscrete   =         2
 Nquad       =        -1
 iMpol       =        -4
 iNtor       =        -4
 Lsparse     =         0
 Lsvdiota    =         0
 imethod     =         3
 iorder      =         2
 iprecon     =         1
 iotatol     =  -1.000000000000000E+00
/
&locallist
 LBeltrami   =         4
 Linitgues   =         1
 Lposdef     =         0
 NiterGMRES  =       500
 epsgmres     =       1e-14
/
&globallist
 Lfindzero   =         2
 escale      =   0.000000000000000E+00
 pcondense   =   4.000000000000000E+00
 forcetol    =   1.000000000000000E-08
 c05xtol     =   1.000000000000000E-12
 c05factor   =   1.000000000000000E-04
 LreadGF     =         F
 opsilon     =   1.000000000000000E+00
 epsilon     =   1.000000000000000E+00
 upsilon     =   1.000000000000000E+00
/
&diagnosticslist
 odetol      =   1.000000000000000E-07
 absreq      =   1.000000000000000E-08
 relreq      =   1.000000000000000E-08
 absacc      =   1.000000000000000E-04
 epsr        =   1.000000000000000E-08
 nPpts       =        101
 nPtrj       =         10
 LHevalues   =         F
 LHevectors  =         F
 LHmatrix    =         T
 Lperturbed  =         0
 dpp         =        -1
 dqq         =        -1
 Lcheck      =         1
 Ltiming     =         F
/
&screenlist
 Wpp00aa = T
/
""")

    # 4. Write to output file
    print("Step 4: Writing to output file...")
    with open(output_spec_file, 'w') as f:
        f.write("\n".join(spec_content))
    print(f"--- Success! ---")
    print(f"Generated SPEC input file: {output_spec_file}")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("\nUsage:\t  python convert_vmec_to_spec.py <vmec_input_file> <vmec_wout_file> <output_spec_file> <num_shells>")
        print("Example:\t  python convert_vmec_to_spec.py input.hs wout_hs.nc hs_10vol.sp 10\n")
        sys.exit(1)

    vmec_in_arg = sys.argv[1]
    wout_arg = sys.argv[2]
    spec_out_arg = sys.argv[3]
    try:
        num_shells_arg = int(sys.argv[4])
        if num_shells_arg <= 0:
            print("Error: <num_shells> must be a positive integer.")
            sys.exit(1)
    except ValueError:
        print("Error: <num_shells> must be an integer.")
        sys.exit(1)
    
    main_converter(vmec_in_arg, wout_arg, spec_out_arg, num_shells_arg)