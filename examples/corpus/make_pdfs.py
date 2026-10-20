"""Build the example corpus's PDF manuals: python examples/corpus/make_pdfs.py

The P-200 manual exists in two revisions, so the corpus exercises what real PDFs need:
numbered headings in larger type, ruled tables, a running header and footer on every page
(removed at ingestion), a scanned annex page with no text layer (reported, and OCR'd only
with DOCURAG_OCR_BACKEND=azure-di), and an older revision whose values must never be
answered once revision C is indexed. The documents are fictional, written for this repository.
"""

from __future__ import annotations

import io
from pathlib import Path

import pymupdf

HERE = Path(__file__).parent
PAGE = pymupdf.paper_rect("a4")
CONTENT = pymupdf.Rect(56, 76, PAGE.width - 56, PAGE.height - 76)
DATES = {"B": "D:20230315000000", "C": "D:20250601000000"}

CSS = """
* { font-family: sans-serif; }
body { font-size: 10pt; line-height: 1.35; }
h1 { font-size: 17pt; margin: 0 0 8pt 0; }
h2 { font-size: 13.5pt; margin: 12pt 0 4pt 0; }
h3 { font-size: 11.5pt; margin: 8pt 0 3pt 0; }
p, li { margin: 0 0 4pt 0; }
table { border-collapse: collapse; margin: 4pt 0 8pt 0; }
th, td { border: 0.8pt solid black; padding: 2pt 6pt; text-align: left; }
th { font-weight: bold; }
"""


def manual_html(rev: str) -> str:
    """Revision C is current; revision B carries the older values the eval must not return."""
    c = rev == "C"
    carter = "49" if c else "45"
    grease_hours = "2 000" if c else "3 000"
    seal_hours = "16 000" if c else "12 000"
    grease_dc, grease_opp = ("12", "8") if c else ("15", "10")
    supersedes = (
        "<p>Ce manuel s'applique aux pompes centrifuges P-200 installées sur les circuits de "
        "refroidissement. Il remplace la révision B. Les valeurs de ce manuel priment sur toute "
        "fiche antérieure.</p>"
        if c
        else "<p>Ce manuel s'applique aux pompes centrifuges P-200 installées sur les circuits de "
        "refroidissement. Première diffusion après la mise en service du site.</p>"
    )
    annex_note = (
        "<p>La fiche de relevé vibratoire figure en annexe A (page numérisée).</p>" if c else ""
    )
    return f"""
<h1>Manuel de maintenance — Pompe centrifuge P-200 (Rév. {rev})</h1>

<h2>1 Généralités</h2>
{supersedes}
<p>Les opérations décrites ici sont réservées au personnel de maintenance habilité. Toute
opération non décrite doit faire l'objet d'une demande au service méthodes.</p>
{annex_note}

<h2>2 Sécurité</h2>
<h3>2.1 Consignation</h3>
<p>Avant toute intervention, consigner électriquement la pompe (sectionneur Q12 ouvert et
cadenassé) puis hydrauliquement (vannes V1 et V2 fermées, circuit purgé). Vérifier l'absence
de pression au manomètre PI-201 : il doit indiquer 0 bar.</p>
<p>Danger : ne jamais desserrer les brides tant que le circuit est sous pression.</p>
<h3>2.2 Équipements de protection</h3>
<p>Gants anti-coupure, lunettes et chaussures de sécurité. Casque antibruit obligatoire si une
pompe voisine est en service (niveau sonore supérieur à 85 dB(A)).</p>

<h2>3 Maintenance préventive</h2>
<h3>3.1 Périodicités</h3>
<table>
<tr><th>Opération</th><th>Périodicité</th><th>Durée</th></tr>
<tr><td>Contrôle visuel des fuites</td><td>Hebdomadaire</td><td>10 min</td></tr>
<tr><td>Graissage des roulements</td><td>Toutes les {grease_hours} h</td><td>20 min</td></tr>
<tr><td>Mesure vibratoire</td><td>Mensuelle</td><td>15 min</td></tr>
<tr><td>Remplacement de la garniture mécanique</td>
<td>{seal_hours} h ou fuite &gt; 10 gouttes/min</td><td>3 h</td></tr>
</table>
<h3>3.2 Graissage des roulements</h3>
<p>Utiliser une graisse lithium NLGI 2 (référence GR-L2). Quantité : {grease_dc} g côté
accouplement, {grease_opp} g côté opposé. Ne pas sur-graisser : un excès de graisse provoque
un échauffement du palier.</p>

<!-- page -->
<h2>4 Couples de serrage</h2>
<p>Serrer à froid, en croix, en deux passes (50 % puis 100 % du couple).</p>
<table>
<tr><th>Élément</th><th>Vis</th><th>Couple (N·m)</th></tr>
<tr><td>Brides d'aspiration</td><td>M16</td><td>120</td></tr>
<tr><td>Brides de refoulement</td><td>M12</td><td>60</td></tr>
<tr><td>Carter de pompe</td><td>M10</td><td>{carter}</td></tr>
<tr><td>Pattes moteur</td><td>M12</td><td>70</td></tr>
</table>

<h2>5 Dépannage</h2>
<h3>5.1 Codes alarme</h3>
<table>
<tr><th>Code</th><th>Signification</th><th>Action</th></tr>
<tr><td>E-12</td><td>Température palier &gt; 85 °C</td><td>Vérifier le graissage (§ 3.2)</td></tr>
<tr><td>E-41</td><td>Vibration &gt; 7,1 mm/s</td><td>Contrôler l'alignement, puis les roulements</td></tr>
<tr><td>E-57</td><td>Pression d'aspiration basse</td><td>Vérifier la vanne V1 et le filtre F-3</td></tr>
</table>
<h3>5.2 Vibrations excessives (alarme E-41)</h3>
<ol>
<li>Consigner la pompe (§ 2.1).</li>
<li>Contrôler l'alignement moteur-pompe au laser : tolérance 0,05 mm.</li>
<li>Si l'alignement est correct, mesurer le spectre vibratoire : une raie à la fréquence de
défaut de bague extérieure indique un roulement endommagé.</li>
<li>Remplacer les roulements 6309-2Z (côté accouplement) et 6307-2Z (côté opposé).</li>
</ol>
<h3>5.3 Pression d'aspiration basse (alarme E-57)</h3>
<p>Vérifier que la vanne V1 est entièrement ouverte, puis nettoyer le filtre d'aspiration F-3
(élément filtrant FE-3). Contrôler enfin le niveau de la bâche d'aspiration : il doit rester
supérieur à 1,2 m pour éviter la cavitation.</p>

<!-- page -->
<h2>6 Caractéristiques techniques</h2>
<table>
<tr><th>Caractéristique</th><th>Valeur</th></tr>
<tr><td>Débit nominal</td><td>120 m³/h</td></tr>
<tr><td>Hauteur manométrique totale</td><td>45 m</td></tr>
<tr><td>Vitesse de rotation</td><td>2 950 tr/min</td></tr>
<tr><td>Puissance du moteur</td><td>22 kW</td></tr>
<tr><td>Masse de l'ensemble</td><td>310 kg</td></tr>
</table>

<h2>7 Pièces de rechange</h2>
<table>
<tr><th>Désignation</th><th>Référence</th><th>Quantité par pompe</th></tr>
<tr><td>Garniture mécanique</td><td>GM-45</td><td>1</td></tr>
<tr><td>Roulement côté accouplement</td><td>6309-2Z</td><td>1</td></tr>
<tr><td>Roulement côté opposé</td><td>6307-2Z</td><td>1</td></tr>
<tr><td>Joint torique de chemise</td><td>JT-80x3</td><td>2</td></tr>
<tr><td>Élément filtrant d'aspiration</td><td>FE-3</td><td>1</td></tr>
</table>
"""


ANNEX_HTML = """
<h2>Annexe A — Fiche de relevé vibratoire</h2>
<p>Mesurer la vitesse vibratoire efficace sur les deux paliers, dans les trois directions
(horizontale, verticale, axiale) : P1 palier côté accouplement, P2 palier côté opposé.</p>
<table>
<tr><th>Vitesse vibratoire</th><th>État</th><th>Conduite à tenir</th></tr>
<tr><td>Jusqu'à 2,8 mm/s</td><td>Bon</td><td>Relevé mensuel</td></tr>
<tr><td>De 2,8 à 7,1 mm/s</td><td>À surveiller</td>
<td>Relevé hebdomadaire, prévenir le service méthodes</td></tr>
<tr><td>Au-delà de 7,1 mm/s</td><td>Alarme E-41</td><td>Arrêt et diagnostic (§ 5.2)</td></tr>
</table>
<p>Date : ........ Pompe : P-200-.... Technicien : ........ Visa chef d'équipe : ........</p>
"""


def _layout(html: str) -> pymupdf.Document:
    """Lay HTML out over as many A4 pages as it needs."""
    story = pymupdf.Story(html=html, user_css=CSS)
    buffer = io.BytesIO()
    writer = pymupdf.DocumentWriter(buffer)
    more = True
    while more:
        device = writer.begin_page(PAGE)
        more, _ = story.place(CONTENT)
        story.draw(device)
        writer.end_page()
    writer.close()
    return pymupdf.open("pdf", buffer.getvalue())


def _scanned_page(doc: pymupdf.Document, html: str, header: str) -> None:
    """Append the annex as a page image only: printed, then scanned, with no text layer."""
    printed = _layout(html)
    page = printed[0]
    page.insert_text((56, 48), header, fontsize=8.5, fontname="helv")
    pix = page.get_pixmap(dpi=110, colorspace=pymupdf.csGRAY)
    pix.gamma_with(0.85)  # the toner of a photocopy
    scan = doc.new_page(width=PAGE.width, height=PAGE.height)
    scan.insert_image(scan.rect, pixmap=pix)


def build(rev: str) -> Path:
    doc = pymupdf.open()
    for part in manual_html(rev).split("<!-- page -->"):  # each part starts a new page
        doc.insert_pdf(_layout(part))
    header = f"Manuel P-200 - Rév. {rev}"
    if rev == "C":
        _scanned_page(doc, ANNEX_HTML, header)
    total = doc.page_count
    for page in doc:
        if not page.get_text().strip():
            continue  # the scanned page already carries its own printed header
        page.insert_text((56, 48), header, fontsize=8.5, fontname="helv")
        page.insert_text(
            (PAGE.width - 110, PAGE.height - 40),
            f"Page {page.number + 1}/{total}",
            fontsize=8.5,
            fontname="helv",
        )
    doc.set_metadata(
        {
            "title": f"Manuel de maintenance — Pompe centrifuge P-200 (Rév. {rev})",
            "author": "Direction maintenance (document fictif)",
            "creationDate": DATES[rev],
            "modDate": DATES[rev],
        }
    )
    out = HERE / "manuals" / f"pompe_p200_rev_{rev.lower()}.pdf"
    doc.save(out, garbage=4, deflate=True, no_new_id=True)
    return out


if __name__ == "__main__":
    for revision in ("B", "C"):
        print(build(revision))
