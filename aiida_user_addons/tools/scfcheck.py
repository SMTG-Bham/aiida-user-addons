"""
Retrospectively check the SCF cycles.
VASP does not raise error when the electronic structure is not converged,
even when the NELM is reached, but instead pretend everthing is OK.
Here we retrospectively check if NELM has ever being reached
"""

import io
import re
from gzip import GzipFile

from tqdm.auto import tqdm

RE_ITER = re.compile(r"-+ Iteration +(\d+)\( *(\d+)\)")
RE_NELM = re.compile(r" +NELM += *(\d+);")


def check_outcar(content):
    """
    Detect SCF issue in the OUTCAR.

    Track the number of electronic iterations for each ionic step.

    Args:
        content: A file handle that can be iterated

    Returns:
        a tuple of (is_ok, problem_steps). The latter is a list of
        ionic steps that the electronic structure is not converged
    """
    NELM = -999
    problems = []
    for line in content:
        if NELM < 0:
            if "NELM" in line:
                match = RE_NELM.match(line)
                if match:
                    NELM = int(match.group(1))
                continue

        match = RE_ITER.match(line)
        if match:
            ionic, electronic = map(int, match.groups())
            if electronic >= NELM:
                problems.append(ionic)

    # Unreliable if the last ionic steps has breached the NELM
    if problems and problems[-1] == ionic:
        is_ok = False
    else:
        is_ok = True

    return is_ok, problems


def database_sweep():
    """
    Perform a database sweep to detect problematic CalcJob
    The detected jobs will be set an extra tag {"nelm_break": True}
    """
    from aiida.orm import QueryBuilder
    from aiida.plugins import CalculationFactory, DataFactory

    Vasp = CalculationFactory("vasp.vasp")
    Folder = DataFactory("folder")

    q = QueryBuilder()
    q.append(
        Vasp,
        filters={"attributes.exit_status": 0, "extras": {"!has_key": "nelm_breach"}},
        project=["label", "uuid", "*"],
    )
    q.append(Folder, project=["*"])

    not_converged = []
    no_data = []
    for label, uuid, calc, retrieved in tqdm(q.iterall(), total=q.count()):
        fnames = retrieved.list_object_names()
        if "OUTCAR" in fnames:
            with retrieved.open("OUTCAR") as fhandle:
                res = check_outcar(fhandle)
        elif "OUTCAR.gz" in fnames:
            with retrieved.open("OUTCAR.gz", mode="rb") as fhandle:
                with GzipFile(fileobj=fhandle, mode="r") as fhandle_:
                    wrapper = io.TextIOWrapper(fhandle_)
                    res = check_outcar(wrapper)
        else:
            no_data.append((label, uuid))
        if res[0] is False:
            calc.set_extra("nelm_breach", True)
            not_converged.append((label, uuid))
        else:
            calc.set_extra("nelm_breach", False)

    for label, uuid in not_converged:
        print(f"{label} {uuid} is not electronically converged!!")

    for label, uuid in no_data:
        print(f"{label} {uuid} does not have any OUTCAR to check for!!")
