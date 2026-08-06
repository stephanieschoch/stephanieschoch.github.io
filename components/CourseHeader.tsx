"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";

interface NavLink {
  href: string;
  label: string;
}

interface CourseHeaderProps {
  courseName: string;
  basePath: string;
  navLinks: NavLink[];
}

export default function CourseHeader({ courseName, basePath, navLinks }: CourseHeaderProps) {
  const pathname = usePathname();
  const [menuOpen, setMenuOpen] = useState(false);

  const isActive = (href: string) => {
    if (href.includes("#")) return false;
    return pathname === href;
  };

  const handleClick = (e: React.MouseEvent<HTMLAnchorElement>, href: string) => {
    const hashIndex = href.indexOf("#");
    if (hashIndex !== -1) {
      const basePage = href.substring(0, hashIndex) || pathname;
      const hash = href.substring(hashIndex + 1);
      if (basePage === pathname) {
        e.preventDefault();
        const el = document.getElementById(hash);
        if (el) {
          el.scrollIntoView({ behavior: "smooth" });
        }
      }
    } else if (href === pathname) {
      e.preventDefault();
      window.scrollTo({ top: 0, behavior: "smooth" });
    }
  };

  return (
    <header className="bg-[#f0f0f0] sticky top-0 z-50">
      <nav className="max-w-4xl mx-auto px-6 py-4 flex items-center justify-between">
        <Link href={basePath} onClick={(e) => handleClick(e, basePath)} className="text-xl font-bold text-text hover:text-accent no-underline">
          {courseName}
        </Link>

        {/* Mobile menu button */}
        <button
          className="md:hidden text-text"
          onClick={() => setMenuOpen(!menuOpen)}
          aria-label="Toggle menu"
        >
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            {menuOpen ? (
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            ) : (
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            )}
          </svg>
        </button>

        {/* Desktop nav */}
        <ul className="hidden md:flex gap-6">
          {navLinks.map((link) => (
            <li key={link.href}>
              <Link
                href={link.href}
                onClick={(e) => handleClick(e, link.href)}
                className={`no-underline transition-colors ${
                  isActive(link.href)
                    ? "text-text font-bold"
                    : "text-text hover:text-accent"
                }`}
              >
                {link.label}
              </Link>
            </li>
          ))}
        </ul>
      </nav>

      {/* Mobile nav */}
      {menuOpen && (
        <ul className="md:hidden px-6 pb-4 flex flex-col gap-2">
          {navLinks.map((link) => (
            <li key={link.href}>
              <Link
                href={link.href}
                onClick={(e) => { handleClick(e, link.href); setMenuOpen(false); }}
                className={`no-underline block py-1 ${
                  isActive(link.href)
                    ? "text-text font-bold"
                    : "text-text hover:text-accent"
                }`}
              >
                {link.label}
              </Link>
            </li>
          ))}
        </ul>
      )}
    </header>
  );
}
