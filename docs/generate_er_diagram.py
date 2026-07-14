"""Generates the ER diagram (docs/er_diagram.png) for the Credit Card
Approval Prediction System, matching the five entities described in the
project brief."""

import graphviz

g = graphviz.Digraph("ER", format="png")
g.attr(rankdir="LR", bgcolor="white", fontname="Helvetica", splines="ortho")
g.attr("node", shape="none", fontname="Helvetica", fontsize="11")

def entity(name, pk, attrs):
    rows = "".join(
        f'<TR><TD ALIGN="LEFT" PORT="{a.split()[0]}"><FONT POINT-SIZE="11">{a}</FONT></TD></TR>'
        for a in attrs
    )
    label = f'''<
    <TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" CELLPADDING="6">
      <TR><TD BGCOLOR="#10162B"><FONT COLOR="white"><B>{name}</B></FONT></TD></TR>
      <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="11"><B>{pk} (PK)</B></FONT></TD></TR>
      {rows}
    </TABLE>>'''
    g.node(name, label=label)

entity("Users", "UserID", ["Name", "Email", "Password", "Role"])
entity("Applicant_Details", "ApplicantID",
       ["UserID (FK)", "IncomeType", "EducationType", "FamilyStatus",
        "HousingType", "EmploymentDays"])
entity("Credit_History", "HistoryID",
       ["ApplicantID (FK)", "PastDefaults", "OverdueMonths",
        "CreditUtilization", "ExistingCreditLines"])
entity("ML_Model", "ModelID", ["ModelName", "Algorithm", "TrainedDate", "Accuracy"])
entity("Approval_Prediction", "PredictionID",
       ["ApplicantID (FK)", "ModelID (FK)", "Result", "Probability", "PredictionDate"])

g.attr("edge", fontname="Helvetica", fontsize="10", color="#4A5170", arrowsize="0.7")
g.edge("Users", "Applicant_Details", label="1 to Many")
g.edge("Applicant_Details", "Credit_History", label="1 to Many")
g.edge("Applicant_Details", "Approval_Prediction", label="1 to 1")
g.edge("ML_Model", "Approval_Prediction", label="1 to Many")
g.edge("Credit_History", "Approval_Prediction", label="informs")

g.render("/home/claude/credit_card_approval_prediction/docs/er_diagram", cleanup=True)
print("Saved ER diagram to docs/er_diagram.png")
