import React from "react";
import Layout from "../components/layout";
import { Link } from "gatsby";

const AboutMe = () => {
  return (
    <Layout>
      <h1>Hello there!</h1>
      <p>
        Thanks for checking out my website. I'm <b>Bruno Opsenica</b> and I've recently decided to try and make my hobby
        of learning about 3D programming and computer graphics into a career. In order to do that, I've created this
        blog as a way of chronicling the things I'm learning about. It's meant to be a mechanism to make my learning
        visible to others, get feedback and (eventually, hopefully) provide help to others if they decide to go down the
        same path. You can see examples of my work on <Link to="/projects">the portfolio page</Link> or in{" "}
        <a href="https://github.com/bruop/">my Github profile.</a>
      </p>
      <p>
        I live in Toronto, and work as a software engineer at a startup called <a href="https://delphia.com">Delphia</a>
        . Before that, I worked at the direct precursor of Delphia, <a href="https://voxpoplabs.com">Vox Pop Labs</a>,
        where I got an opportunity to build a few different interactive data visualizations, like{" "}
        <a href="http://votecompass.com">Vote Compass</a> and <a href="https://echoes.cbc.ca">Echoes</a>
        <sup>1</sup>.
      </p>

      <p>
        Before that, I attended the University of Toronto where I specialized in Physics and graduated in 2012. It was
        while taking courses on fluid mechanics and geophysics that I discovered my love of programming, and a fourth
        year course in computer graphics sealed my fate.
      </p>

      <p>
        I'm currently looking for work as a graphics programmer! If you are interested, you can{" "}
        <a href="mailto:bruno.opsenica@gmail.com">contact me directly</a>, or download a{" "}
        <a href="/resume.pdf">copy of my resume.</a>
      </p>

      <hr />

      <h4>Notes:</h4>
      <p>
        1: Not sure how long this link will actually stay up! The project launched in late 2017, but appears to still be
        available.
      </p>
    </Layout>
  );
};

export default AboutMe;
