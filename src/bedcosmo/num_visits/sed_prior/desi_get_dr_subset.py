#!/usr/bin/env python
"""
Script to download a tiny subset of DESI data, finding healpix with most targets near a position.
"""

import os
import argparse
from pathlib import Path
import shutil
import numpy as np
import fitsio
from astropy.table import Table
import requests
import urllib3
from urllib.parse import urljoin
import re
from html.parser import HTMLParser
from tqdm import tqdm

urllib3.disable_warnings()

class Settings:
    DESI_USER = None
    DESI_PASSWD = None

# Dictionary mapping data releases to their primary specprod
DR_SPECPROD_MAP = {
    'edr': 'fuji',
    'dr1': 'iron', 
    'dr2': 'loa'
}

# Dictionary mapping data releases to their catalog path structure
DR_CATALOG_PATH = {
    'edr': 'zcatalog',      # EDR has no v1 subdirectory
    'dr1': 'zcatalog/v1',
    'dr2': 'zcatalog/v1'
}

# Dictionary mapping data releases to their healpix survey/program path
DR_HEALPIX_PATH = {
    'edr': 'sv3',          # EDR uses sv3/dark
    'dr1': 'main',         # DR1/DR2 use main/dark
    'dr2': 'main'
}

class DirListParser(HTMLParser):
    """Simple HTML parser to extract directory listing links"""
    def __init__(self):
        super().__init__()
        self.links = []
        
    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            for name, value in attrs:
                if name == 'href':
                    self.links.append(value)

def get_desi_login_password():
    """Get DESI login credentials from ~/.desi_http_user file"""
    if Settings.DESI_USER is None:
        config = os.path.join(os.environ['HOME'], '.desi_http_user')
        if not os.path.exists(config):
            raise Exception('''You need to specify the DESI_USER/DESI_PASSWD.
Put them in $HOME/.desi_http_user like that:
username:password
''')
        user, pwd = open(config).read().rstrip().split(':')
        Settings.DESI_USER, Settings.DESI_PASSWD = user, pwd
    return Settings.DESI_USER, Settings.DESI_PASSWD

def top_healpix_from_catalog(catalog_file, n, exclude=None):
    """Return the *n* HEALPix IDs with the most catalog rows (global DR1 counts)."""
    exclude = set(exclude or [])
    data = fitsio.read(catalog_file, columns=["HEALPIX"], ext=1)
    unique_pix, counts = np.unique(data["HEALPIX"], return_counts=True)
    order = np.argsort(counts)[::-1]
    selected = []
    for idx in order:
        pix = int(unique_pix[idx])
        if pix in exclude:
            continue
        selected.append(pix)
        if len(selected) >= n:
            break
    return selected


def healpix_coadd_path(local_base_path, specprod, data_release, healpix_id):
    """Local path to coadd FITS for a HEALPix patch (used to detect existing downloads)."""
    healpix_survey = get_healpix_survey_path(data_release)
    prefix, healpix_path = get_healpix_path(healpix_id)
    return os.path.join(
        local_base_path,
        f"spectro/redux/{specprod}/healpix/{healpix_survey}/dark/{healpix_path}",
        f"coadd-{healpix_survey}-dark-{healpix_id}.fits",
    )


def find_best_healpix(catalog_file, center_ra, center_dec, radius=0.5):
    """
    Find healpix with most targets within radius of center position
    
    Parameters:
        catalog_file (str): Path to zall-pix-{specprod}.fits catalog
        center_ra (float): Center right ascension in degrees
        center_dec (float): Center declination in degrees
        radius (float): Search radius in degrees
        
    Returns:
        int: Healpix ID with most targets in search area
    """
    # Read relevant columns from catalog
    data = fitsio.read(catalog_file, columns=['TARGET_RA', 'TARGET_DEC', 'HEALPIX'], ext=1)
    
    # Convert coordinates to radians for spherical geometry
    ra_rad = np.radians(data['TARGET_RA'])
    dec_rad = np.radians(data['TARGET_DEC'])
    center_ra_rad = np.radians(center_ra)
    center_dec_rad = np.radians(center_dec)
    
    # Haversine formula for angular separation
    dlon = ra_rad - center_ra_rad
    dlat = dec_rad - center_dec_rad
    a = np.sin(dlat/2)**2 + np.cos(dec_rad) * np.cos(center_dec_rad) * np.sin(dlon/2)**2
    dist_rad = 2 * np.arcsin(np.sqrt(a))
    dist_deg = np.degrees(dist_rad)
    
    # Find objects within search radius
    mask = dist_deg <= radius
    nearby_targets = data[mask]
    
    # Count targets per healpix
    unique_pix, counts = np.unique(nearby_targets['HEALPIX'], return_counts=True)
    
    print("\nHealpix analysis within search radius:")
    for pix, count in sorted(zip(unique_pix, counts)):
        print(f"HEALPIX {pix}: {count} targets")
    
    # Get healpix with most targets
    top_healpix = unique_pix[np.argmax(counts)]
    return top_healpix

def get_healpix_path(healpix_id):
    """Construct the healpix path based on the ID"""
    healpix_str = str(healpix_id)
    prefix = healpix_str if len(healpix_str) < 3 else healpix_str[:3]
    return prefix, f"{prefix}/{healpix_str}"

def list_directory(url, auth=None):
    """List contents of a directory on DESI server using requests"""
    headers = {}
    
    try:
        if auth is not None:
            response = requests.get(url, auth=auth, verify=False)
        else:
            response = requests.get(url, verify=False)
        
        if response.status_code == 200:
            # Parse HTML to get links
            parser = DirListParser()
            parser.feed(response.text)
            
            # Filter out parent directory and create full URLs
            entries = [urljoin(url, link) for link in parser.links 
                      if not link == "../" and not link == "./"]
            
            return entries
        else:
            print(f"Error listing directory {url}: HTTP status {response.status_code}")
            return None
    except Exception as e:
        print(f"Error listing directory {url}: {str(e)}")
        return None

def download_file(url, local_path=None, remote_base_url=None, auth=None, local_base_path=None):
    """
    Download a file from DESI server if it doesn't exist locally
    
    Parameters:
        url (str): URL of the file to download
        local_path (str, optional): Local path to save file. If None and remote_base_url 
                                   is provided, will be constructed from url
        remote_base_url (str, optional): Base URL to calculate relative path
        auth (tuple, optional): (username, password) tuple
        local_base_path (str, optional): Base directory for local file storage
    """
    # Handle path construction if local_path not provided
    if local_path is None:
        if remote_base_url is not None and local_base_path is not None:
            rel_path = url[len(remote_base_url):] if url.startswith(remote_base_url) else os.path.basename(url)
            local_path = os.path.join(local_base_path, rel_path)
        else:
            raise ValueError("Either local_path or both remote_base_url and local_base_path must be provided")
    
    if os.path.exists(local_path):
        print(f"File already exists, skipping: {local_path}")
        return True
        
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    try:
        # Try with provided auth or no auth initially
        response = requests.get(url, auth=auth, stream=True, verify=False)
        response.raise_for_status()
        
        # Use a temporary file to prevent corrupt downloads
        tmpfile = local_path + '.downloading'
        total = int(response.headers.get("content-length", 0))
        chunk_size = 1024 * 1024  # 1 MB chunks

        desc = os.path.basename(local_path)

        with open(tmpfile, "wb") as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=desc,
            disable=(total == 0),
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        # Rename only after successful download
        os.rename(tmpfile, local_path)
        
        print(f"Downloaded{' (with auth)' if auth else ''}: {local_path}")
        return True
    except requests.exceptions.HTTPError as e:
        # If unauthorized and no auth provided, try to get credentials
        if e.response.status_code == 401 and auth is None:
            try:
                user, pwd = get_desi_login_password()
                print("Retrying with authentication...")
                return download_file(url, local_path, remote_base_url, (user, pwd), local_base_path)
            except Exception as auth_e:
                print(f"Error with credentials: {str(auth_e)}")
                return False
        print(f"Error downloading {url}: {str(e)}")
        return False
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        return False

def download_directory(url, local_base_path, remote_base_url, auth=None):
    """Download all files in a directory"""
    contents = list_directory(url, auth)
    
    if contents is None:
        if auth is None:
            try:
                user, pwd = get_desi_login_password()
                print("Retrying directory listing with authentication...")
                return download_directory(url, local_base_path, remote_base_url, (user, pwd))
            except Exception as e:
                print(f"Error with credentials: {str(e)}")
                return False
        return False
    
    success = True
    for item_url in contents:
        # Skip directories
        if item_url.endswith('/'):
            continue
        
        # Use the consolidated download_file function
        success &= download_file(item_url, remote_base_url=remote_base_url, 
                               local_base_path=local_base_path, auth=auth)
    
    return success

def get_tile_date(remote_base_url, tileid, specprod, auth=None):
    """Get the most recent date directory for a tile"""
    tile_url = f"{remote_base_url}spectro/redux/{specprod}/tiles/cumulative/{tileid}/"
    contents = list_directory(tile_url, auth)
    
    if contents is None:
        if auth is None:
            try:
                user, pwd = get_desi_login_password()
                return get_tile_date(remote_base_url, tileid, specprod, (user, pwd))
            except Exception:
                return None
        return None
    
    # Filter for date-like directories (8 digits) and get the latest
    date_dirs = []
    for d in contents:
        if d.endswith('/'):
            # Extract the directory name and check if it's 8 digits
            dirname = os.path.basename(d.rstrip('/'))
            if dirname.isdigit() and len(dirname) == 8:
                date_dirs.append(dirname)
    
    return max(date_dirs) if date_dirs else None

def analyze_tiles(redrock_file):
    """
    Analyze tileIDs from redrock file and return top N tiles by target count
    
    Parameters:
        redrock_file (str): Path to redrock FITS file
    """
    try:
        r = Table.read(redrock_file, hdu=3)
        unique_values, counts = np.unique(r['TILEID'], return_counts=True)
        
        # Sort tiles by count in descending order
        sorted_indices = np.argsort(counts)[::-1]
        sorted_tileids = unique_values[sorted_indices]
        sorted_counts = counts[sorted_indices]
        
        print("\nTile analysis (sorted by target count):")
        for tileid, count in zip(sorted_tileids, sorted_counts):
            print(f"TILEID {tileid}: {count} targets")

        return sorted_tileids
    except Exception as e:
        print(f"Error analyzing redrock file: {str(e)}")
        return None

def get_base_url(data_release=None, specprod=None):
    """
    Determine the appropriate base URL based on data release and specprod.
    
    Parameters:
        data_release (str): Data release identifier (e.g., 'edr', 'dr1', 'dr2')
        specprod (str): Spectroscopic production name (e.g., 'fuji', 'iron', 'loa')
        
    Returns:
        tuple: (base_url, specprod_to_use, requires_auth)
    """
    requires_auth = False
    
    # DR2 is not public yet, use proprietary URL
    if data_release == 'dr2':
        requires_auth = True
        specprod_to_use = specprod or 'loa'
        return 'https://data.desi.lbl.gov/desi/', specprod_to_use, requires_auth
    
    # Case 1: Both data_release and specprod provided
    if data_release and specprod:
        # Check if this is a known DR/specprod combination
        if data_release in DR_SPECPROD_MAP and DR_SPECPROD_MAP[data_release] == specprod:
            base_url = f'https://data.desi.lbl.gov/public/{data_release}/'
            return base_url, specprod, requires_auth
        else:
            # Custom specprod - need authentication
            requires_auth = True
            return 'https://data.desi.lbl.gov/desi/', specprod, requires_auth
    
    # Case 2: Only data_release provided
    elif data_release and not specprod:
        if data_release in DR_SPECPROD_MAP:
            base_url = f'https://data.desi.lbl.gov/public/{data_release}/'
            specprod_to_use = DR_SPECPROD_MAP[data_release]
            return base_url, specprod_to_use, requires_auth
        else:
            print(f"Unknown data release: {data_release}. Using default dr1/iron.")
            base_url = 'https://data.desi.lbl.gov/public/dr1/'
            return base_url, 'iron', requires_auth
    
    # Case 3: Only specprod provided
    elif not data_release and specprod:
        # Check if this specprod is associated with a known DR
        for dr, sp in DR_SPECPROD_MAP.items():
            if sp == specprod:
                # Handle dr2 specially as non-public
                if dr == 'dr2':
                    requires_auth = True
                    return 'https://data.desi.lbl.gov/desi/', specprod, requires_auth
                else:
                    base_url = f'https://data.desi.lbl.gov/public/{dr}/'
                    return base_url, specprod, requires_auth
        
        # If not found in known DRs, use proprietary URL
        requires_auth = True
        return 'https://data.desi.lbl.gov/desi/', specprod, requires_auth
    
    # Case 4: Neither provided - use default dr1/iron
    else:
        base_url = 'https://data.desi.lbl.gov/public/dr1/'
        return base_url, 'iron', requires_auth

def get_catalog_path(data_release, specprod):
    """
    Get the appropriate catalog path based on data release.
    Different releases have different directory structures.
    
    Parameters:
        data_release (str): Data release identifier (e.g., 'edr', 'dr1', 'dr2')
        specprod (str): Spectroscopic production name
        
    Returns:
        str: Catalog path for the given data release
    """
    # Use the mapping or default to zcatalog/v1
    return DR_CATALOG_PATH.get(data_release, 'zcatalog/v1')

def get_healpix_survey_path(data_release):
    """
    Get the appropriate healpix survey path based on data release.
    Different releases use different survey/program paths.
    
    Parameters:
        data_release (str): Data release identifier (e.g., 'edr', 'dr1', 'dr2')
        
    Returns:
        str: Healpix survey path for the given data release
    """
    # Use the mapping or default to main
    return DR_HEALPIX_PATH.get(data_release, 'main')

def main():
    """
    Main function to run the DESI data downloader.
    Default coordinates:
    - For dr1/dr2: (RA=55, Dec=-9), which typically results in downloading healpix 23040 for tutorial purposes.
    - For edr: (RA=179.6, Dec=0.0), which typically results in downloading healpix 26965 and 12 tiles
      corresponding to Rosette 1 in the EDR paper (overlapping with GAMA G12 and KiDS-N).
    """
    parser = argparse.ArgumentParser(
        description='Download DESI data for targets near a position. The default coordinates '
                    'depend on data release: for dr1/dr2, RA=55, Dec=-9.0 (healpix 23040); '
                    'for edr, RA=179.6, Dec=0.0 (healpix 26965, Rosette 1 field overlapping with GAMA G12 and KiDS-N).')
    
    # We'll set the default RA/Dec based on data release later
    parser.add_argument('--ra', type=float, 
                        help='Center right ascension in degrees (default depends on data release)')
    parser.add_argument('--dec', type=float, 
                        help='Center declination in degrees (default depends on data release)')
    parser.add_argument('--radius', type=float, default=0.1, help='Search radius in degrees (default: 0.1)')
    parser.add_argument('--dr', default='dr1', help='Data release (e.g., edr, dr1, dr2). Default: dr1')
    parser.add_argument('--specprod', help='Spectroscopic production name (e.g., fuji, iron, loa)')
    parser.add_argument('--no-tiles', action='store_true', help='Download only healpix data, skip tile data')
    parser.add_argument(
        '--healpix',
        type=int,
        nargs='+',
        help='Explicit HEALPix ID(s) to download (skips RA/Dec search).',
    )
    parser.add_argument(
        '--top-n-healpix',
        type=int,
        metavar='N',
        help='Download the N densest HEALPix patches from zall-pix (by catalog row count).',
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        default=True,
        help='Skip HEALPix patches whose coadd file already exists locally (default: on).',
    )
    parser.add_argument(
        '--no-skip-existing',
        dest='skip_existing',
        action='store_false',
        help='Re-download HEALPix patches even if coadd exists.',
    )
    parser.add_argument(
        '--skip-catalog',
        action='store_true',
        help='Do not download zall-pix / tile CSVs if catalog FITS already exists.',
    )
    
    # Default output directory based on data release
    default_dir = lambda dr: f"./tiny_{dr.lower()}"
    parser.add_argument('--base-dir', help=f'Base directory for downloads (default: depends on data release, e.g., ./tiny_dr1)')
    args = parser.parse_args()
    # No need to validate dr/specprod since dr has a default value
    
    # Set default RA/Dec based on data release
    if args.dr == 'edr':
        # Set edr-specific defaults if not provided
        if args.ra is None:
            args.ra = 179.6
        if args.dec is None:
            args.dec = 0.0
    else:
        # Set dr1/dr2 defaults if not provided
        if args.ra is None:
            args.ra = 56.0
        if args.dec is None:
            args.dec = -9.0
    
    # Determine the appropriate base URL and specprod
    remote_base_url, specprod, requires_auth = get_base_url(args.dr, args.specprod)
    
    # Set base directory if not explicitly provided
    if args.base_dir is None:
        if args.dr:
            local_base_path = f"./tiny_{args.dr.lower()}"
        else:
            # If only specprod is provided, use it for the directory name
            local_base_path = f"./tiny_{specprod.lower()}"
    else:
        local_base_path = args.base_dir
    
    auth = None
    if requires_auth:
        try:
            user, pwd = get_desi_login_password()
            auth = (user, pwd)
            print(f"Using authenticated access for {specprod}")
        except Exception as e:
            print(f"Error getting credentials: {str(e)}")
            return
    
    print(f"Starting downloads from {remote_base_url}")
    print(f"Using spectroscopic production: {specprod}")
    print(f"Files will be saved to {local_base_path}")
    use_position_search = args.healpix is None and args.top_n_healpix is None
    if use_position_search:
        print(
            f"\nSearching for targets around RA={args.ra}, Dec={args.dec} "
            f"with radius={args.radius} degrees"
        )
        if args.dr == 'edr':
            print("(Default coordinates for EDR are RA=179.6, Dec=0.0, retrieving healpix 26965)")
        else:
            print("(Default coordinates for DR1/DR2 retrieve healpix 23040)")
    
    # Get the appropriate catalog path based on data release
    catalog_subpath = get_catalog_path(args.dr, specprod)
    catalog_file = os.path.join(
        local_base_path,
        f'spectro/redux/{specprod}/{catalog_subpath}/zall-pix-{specprod}.fits',
    )

    if not (args.skip_catalog and os.path.exists(catalog_file)):
        print("\nDownloading tile and exposure CSV files...")
        tiles_url = f"{remote_base_url}spectro/redux/{specprod}/tiles-{specprod}.csv"
        tiles_file = os.path.join(local_base_path, f'spectro/redux/{specprod}/tiles-{specprod}.csv')
        if not download_file(tiles_url, tiles_file, auth=auth):
            print(f"Warning: Failed to download tiles CSV file: {tiles_url}")

        exposures_url = f"{remote_base_url}spectro/redux/{specprod}/exposures-{specprod}.csv"
        exposures_file = os.path.join(
            local_base_path, f'spectro/redux/{specprod}/exposures-{specprod}.csv'
        )
        if not download_file(exposures_url, exposures_file, auth=auth):
            print(f"Warning: Failed to download exposures CSV file: {exposures_url}")

        print("\nDownloading redshift catalog...")
        catalog_url = (
            f"{remote_base_url}spectro/redux/{specprod}/{catalog_subpath}/zall-pix-{specprod}.fits"
        )
        if not download_file(catalog_url, catalog_file, auth=auth):
            print("Failed to download redshift catalog. Cannot continue.")
            return
    elif not os.path.exists(catalog_file):
        print(f"Catalog not found at {catalog_file}; cannot continue.")
        return
    else:
        print(f"\nUsing existing catalog: {catalog_file}")

    if args.healpix is not None:
        healpix_ids = list(args.healpix)
        print(f"\nHEALPix list from --healpix: {healpix_ids}")
    elif args.top_n_healpix is not None:
        healpix_ids = top_healpix_from_catalog(catalog_file, args.top_n_healpix)
        print(f"\nTop {args.top_n_healpix} HEALPix by catalog count: {healpix_ids}")
    else:
        healpix_id = find_best_healpix(catalog_file, args.ra, args.dec, args.radius)
        print(f"\nSelected HEALPIX {healpix_id} with most targets in search region")
        healpix_ids = [healpix_id]

    healpix_survey = get_healpix_survey_path(args.dr)
    success = True
    for healpix_id in healpix_ids:
        coadd_local = healpix_coadd_path(local_base_path, specprod, args.dr, healpix_id)
        if args.skip_existing and os.path.exists(coadd_local):
            print(f"\nSkipping HEALPIX {healpix_id} (coadd exists): {coadd_local}")
            continue

        print(f"\nDownloading files for healpix {healpix_id}...")
        prefix, healpix_path = get_healpix_path(healpix_id)
        healpix_url = (
            f"{remote_base_url}spectro/redux/{specprod}/healpix/"
            f"{healpix_survey}/dark/{healpix_path}/"
        )
        success &= download_directory(healpix_url, local_base_path, remote_base_url, auth)

    if args.no_tiles:
        print("\nSkipping tile data downloads (--no-tiles option specified)")
    elif len(healpix_ids) == 1:
        healpix_id = healpix_ids[0]
        prefix, healpix_path = get_healpix_path(healpix_id)
        redrock_file = os.path.join(
            local_base_path,
            f'spectro/redux/{specprod}/healpix/{healpix_survey}/dark/{healpix_path}',
            f'redrock-{healpix_survey}-dark-{healpix_id}.fits',
        )
        print("\nAnalyzing redrock file for tileIDs...")
        tileids = analyze_tiles(redrock_file)
        if tileids is not None:
            print(f"\nPreparing to download data for {len(tileids)} tiles...")
            print("\nDownloading tile data...")
            for tileid in tileids:
                print(f"\nProcessing TILEID {tileid}...")
                date = get_tile_date(remote_base_url, tileid, specprod, auth)
                if date is None:
                    print(f"Could not find date directory for tile {tileid}")
                    continue
                print(f"Found date directory: {date}")
                tile_url = (
                    f"{remote_base_url}spectro/redux/{specprod}/tiles/cumulative/"
                    f"{tileid}/{date}/"
                )
                success &= download_directory(tile_url, local_base_path, remote_base_url, auth)
    
    if success:
        print("\nAll downloads completed successfully!")
    else:
        print("\nDownloads completed with some errors. Please check the messages above.")

if __name__ == "__main__":
    main()