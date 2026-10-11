"""Keep every user guide clickable from README and connected to the doc index."""
from collections import Counter
from pathlib import Path
import re
import unittest
from urllib.parse import unquote, urlsplit


ROOT=Path(__file__).resolve().parents[1]
DOCS=sorted(set(ROOT.glob('*.md')) | set(ROOT.glob('*.MD')) | {ROOT/'tests/README.md'})


def prose(path):
    return re.sub(r'```[^\n]*\n[\s\S]*?```','',path.read_text())


def links(path):
    for match in re.finditer(r'(?<!!)\[[^\]\n]*\]\(([^)\n]+)\)',prose(path)):
        target=match[1].strip()
        if target.startswith('<'):target=target[1:target.index('>')]
        else:target=target.split()[0]
        url=urlsplit(target)
        if url.scheme or url.netloc:continue
        destination=(path.parent/unquote(url.path)).resolve() if url.path else path.resolve()
        if destination.suffix.lower()=='.md':yield destination,unquote(url.fragment)


def anchors(path):
    found=set()
    used=Counter()
    for line in prose(path).splitlines():
        heading=re.match(r'^#{1,6}\s+(.+?)\s*#*\s*$',line)
        if not heading:continue
        label=re.sub(r'\[([^]]+)\]\([^)]*\)',r'\1',heading[1])
        label=re.sub(r'<[^>]+>','',label).lower()
        slug=re.sub(r'[^\w\s-]','',label).replace(' ','-')
        count=used[slug];used[slug]+=1
        found.add(slug if count==0 else f'{slug}-{count}')
    found.update(re.findall(r'<a\s+(?:id|name)=["\']([^"\']+)',prose(path)))
    return found


class TestDocumentationLinks(unittest.TestCase):
    def test_local_markdown_targets_and_anchors_exist(self):
        for document in DOCS:
            for target,fragment in links(document):
                with self.subTest(document=str(document.relative_to(ROOT)),target=str(target),fragment=fragment):
                    self.assertTrue(target.is_file(),f'Missing Markdown target: {target}')
                    if fragment:self.assertIn(fragment,anchors(target),f'Missing heading anchor: {target}#{fragment}')

    def test_every_guide_is_directly_linked_from_readme(self):
        readme=ROOT/'README.MD'
        destinations={target for target,_ in links(readme)}
        for document in DOCS:
            if document!=readme:
                with self.subTest(document=document.name):self.assertIn(document.resolve(),destinations)

    def test_every_guide_links_back_to_documentation_index(self):
        readme=ROOT/'README.MD'
        for document in DOCS:
            if document!=readme:
                with self.subTest(document=document.name):self.assertIn((readme.resolve(),'documentation'),set(links(document)))


if __name__=='__main__':unittest.main()
